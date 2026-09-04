#!/usr/bin/env python
#train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import sys
import os
import time
from tqdm import tqdm
from .model import Model, sup_contrasitive_loss
from .data import Dataset, collate_fn
from .utils import (get_config, calculate_metric, analyze_key_residues,
                  visualize_attention_heatmap, save_key_residues_report,
                  save_source_code)
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def resolve_ablation_config(config):
    """Apply ablation switches and make the resolved loss weights explicit."""
    defaults = {
        'name': 'B0_full',
        'use_aac_dpc': True,
        'use_cksaap_esmc': True,
        'use_contrastive_loss': True,
        'use_cross_attention': True,
    }
    settings = defaults.copy()
    settings.update(config.get('ablation', {}))
    settings = {
        'name': str(settings['name']),
        'use_aac_dpc': bool(settings['use_aac_dpc']),
        'use_cksaap_esmc': bool(settings['use_cksaap_esmc']),
        'use_contrastive_loss': bool(settings['use_contrastive_loss']),
        'use_cross_attention': bool(settings['use_cross_attention']),
    }
    if not settings['use_aac_dpc'] and not settings['use_cksaap_esmc']:
        raise ValueError(
            "At least one global input branch (AAC-DPC or CKSAAP-ESMC) "
            "must remain enabled"
        )
    if not settings['use_contrastive_loss']:
        config['train']['contrastive_loss_coef'] = 0.0
    if not settings['use_cross_attention']:
        config['train']['interaction_aux_loss_coef'] = 0.0
    config['ablation'] = settings
    return settings


def build_model(config):
    settings = config['ablation']
    return Model(
        embed_hid_dim=config['model']['embed_hid_dim'],
        dropout=config['model']['dropout'],
        interaction_gate_init=config['model'].get(
            'interaction_gate_init', -2.0
        ),
        use_aac_dpc=settings['use_aac_dpc'],
        use_cksaap_esmc=settings['use_cksaap_esmc'],
        use_contrastive_loss=settings['use_contrastive_loss'],
        use_cross_attention=settings['use_cross_attention'],
    )

def predicting(model, loader, save_path=None, threshold=0.5,
               fusion_save_path=None, labels_save_path=None,
               dataset_type="test", collect_attention=False):
    if collect_attention and not model.use_cross_attention:
        raise ValueError(
            "Attention collection was requested for an attention-disabled ablation"
        )
    model.eval()
    device = get_device()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    total_residue_scores1 = []
    total_residue_scores2 = []
    total_cross_attn = []
    total_cross_attn_reverse = []
    total_seqs1 = []
    total_seqs2 = []
    total_fusion_vectors = [] 
    total_protein_ids1 = []
    total_protein_ids2 = []

    print(f'Make prediction for {dataset_type} dataset with {len(loader.dataset)} samples...')
    with torch.no_grad():
        for batch_idx, (ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, y) in enumerate(tqdm(loader)):
            batch_data = [ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, y]
            batch_data = tuple(data.to(device) if isinstance(data, torch.Tensor) else data for data in batch_data)
            ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths, name1, name2, y = batch_data

            out = model(ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths)
            logits = out['logits'].squeeze(-1)
            if not torch.isfinite(logits).all().item():
                bad_indices = torch.where(~torch.isfinite(logits))[0].tolist()
                bad_pairs = [
                    f"{name1[index]}_{name2[index]}"
                    for index in bad_indices
                ]
                raise FloatingPointError(
                    f"Non-finite predictions in {dataset_type} batch "
                    f"{batch_idx + 1}; pairs={bad_pairs}"
                )
            total_preds = torch.cat((total_preds, logits), 0)
            total_labels = torch.cat((total_labels, y.float()), 0)
            residue_scores1, residue_scores2 = out['residue_scores']
            attn_weights = out['attention_weights']
            fusion_vector = out['fusion_vector']  
            total_fusion_vectors.append(fusion_vector)  
            for i in range(len(name1)):
                total_protein_ids1.append(name1[i])
                total_protein_ids2.append(name2[i])
                if collect_attention:
                    seq1 = loader.dataset.get_protein_sequence(
                        name1[i], is_first_protein=True
                    )
                    seq2 = loader.dataset.get_protein_sequence(
                        name2[i], is_first_protein=False
                    )
                    length1 = int(esmc1_lengths[i].item())
                    length2 = int(esmc2_lengths[i].item())
                    if length1 != len(seq1) or length2 != len(seq2):
                        raise ValueError(
                            "Residue output is not aligned to biological "
                            f"sequence: {name1[i]} embedding={length1}, "
                            f"sequence={len(seq1)}; {name2[i]} "
                            f"embedding={length2}, sequence={len(seq2)}"
                        )
                    total_seqs1.append(seq1)
                    total_seqs2.append(seq2)
                    total_residue_scores1.append(
                        residue_scores1[i, :length1].cpu().numpy()
                    )
                    total_residue_scores2.append(
                        residue_scores2[i, :length2].cpu().numpy()
                    )
                    total_cross_attn.append(
                        attn_weights['cross_attn'][i, :length1, :length2]
                        .cpu().numpy()
                    )
                    total_cross_attn_reverse.append(
                        attn_weights['cross_attn_reverse'][
                            i, :length2, :length1
                        ].cpu().numpy()
                    )


    if fusion_save_path:  # Save fusion representations when requested.
        total_fusion_vectors = np.concatenate(total_fusion_vectors, axis=0)  
        fusion_file = fusion_save_path.replace("{dataset_type}", dataset_type)
        np.save(fusion_file, total_fusion_vectors)
    
    if labels_save_path:  # Save labels when requested.
        labels_file = labels_save_path.replace("{dataset_type}", dataset_type)
        np.save(labels_file, total_labels.cpu().numpy())
        
    if save_path:
        with open(save_path, 'w') as f:
            f.write(
                "Protein1_ID\tProtein2_ID\tPredicted_Probability\t"
                "True_Label\tPredicted_Label\n"
            )
            probabilities = total_preds.cpu().numpy().flatten()
            true_labels = total_labels.cpu().numpy().flatten()
            for p1, p2, prob, true_label in zip(
                total_protein_ids1, total_protein_ids2,
                probabilities, true_labels
            ):
                predicted_label = 1 if prob >= threshold else 0
                f.write(
                    f"{p1}\t{p2}\t{prob:.6f}\t{int(true_label)}\t"
                    f"{predicted_label}\n"
                )

    return (total_labels.cpu().numpy().flatten(), 
            total_preds.cpu().numpy().flatten(),
            total_residue_scores1,
            total_residue_scores2,
            total_seqs1,
            total_seqs2,
            {
                'cross_attn': total_cross_attn,
                'cross_attn_reverse': total_cross_attn_reverse
            },
            total_protein_ids1,
            total_protein_ids2
    )

def train(train_args, nn, test_loader, result_dir, threshold=0.5):
    """Train the model"""
    train_losses = []
    train_bce_losses = []
    train_contrastive_losses = []
    train_interaction_aux_losses = []
    train_accs = []
    times = 0.0
    patience = train_args['config']['train']['patience']  
    early_stopping_triggered = False
    attention_model = train_args['model']
    optimizer = train_args['optimizer']
    criterion = train_args["criterion"]
    contrastive_loss_coef = train_args["contrastive_loss_coef"]
    interaction_aux_loss_coef = train_args["interaction_aux_loss_coef"]
    device = get_device()
    attention_model.to(device)
    train_loader = train_args['train_loader']

    best_mcc = -float('inf')
    best_epoch = 0
    metric_mcc = None
    best_val_preds = None
    best_val_labels = None
    best_roc_data = None
    best_prc_data = None
    best_model_path = None

    save_path = os.path.join(result_dir, "predictions.txt")
    
    for epoch in tqdm(range(train_args['epochs'])):
        if early_stopping_triggered:
            print(f"Early stopping triggered at epoch {epoch}")
            with open(train_args['rst_file'], 'a+') as fp:
                fp.write(f"Early stopping triggered at epoch {epoch}\n")
            break

        print(nn, "Running EPOCH", epoch + 1)
        start_time = time.time()
        total_loss = 0
        total_bce_loss = 0
        total_contrastive_loss = 0
        total_interaction_aux_loss = 0
        zero_positive_batches = 0
        correct = 0
        n_batches = 0
        attention_model.train()

        for batch_idx, (ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, labels) in enumerate(tqdm(train_loader)):

            batch_data = [ckesm1, ckesm2, aac1, aac2, dip1, dip2, 
                        esmc1, esmc2, esmc1_lengths, esmc2_lengths, labels]
            batch_data = tuple(data.to(device) if isinstance(data, torch.Tensor) else data for data in batch_data)
            ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths, labels = batch_data
            
            optimizer.zero_grad()
            
            outputs = attention_model(ckesm1, ckesm2, aac1, aac2, dip1, dip2, 
                                    esmc1, esmc2, esmc1_lengths, esmc2_lengths)
            logits = outputs['logits'].squeeze(-1)
            raw_logits = outputs['raw_logits'].squeeze(-1)
            interaction_raw_logits = outputs['interaction_raw_logits'].squeeze(-1)
            seq1_projections, seq2_projections = outputs['projection']
            labels = labels.float()

            if not torch.isfinite(raw_logits).all().item():
                raise FloatingPointError(
                    f"Non-finite main logits at fold={nn}, epoch={epoch + 1}, "
                    f"batch={batch_idx + 1}"
                )
            if (attention_model.use_cross_attention and
                    not torch.isfinite(interaction_raw_logits).all().item()):
                raise FloatingPointError(
                    f"Non-finite interaction logits at fold={nn}, "
                    f"epoch={epoch + 1}, batch={batch_idx + 1}"
                )

            bce_loss = criterion(logits, labels)
            zero_loss = raw_logits.sum() * 0.0
            if interaction_aux_loss_coef > 0:
                interaction_aux_loss = F.binary_cross_entropy_with_logits(
                    interaction_raw_logits, labels
                )
            else:
                interaction_aux_loss = zero_loss
            if contrastive_loss_coef > 0:
                contrastive_loss = sup_contrasitive_loss(
                    seq1_projections, seq2_projections, labels
                )
                if torch.count_nonzero(labels > 0.5).item() == 0:
                    zero_positive_batches += 1
            else:
                contrastive_loss = zero_loss
            loss = (
                bce_loss
                + contrastive_loss_coef * contrastive_loss
                + interaction_aux_loss_coef * interaction_aux_loss
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at fold={nn}, epoch={epoch + 1}, "
                    f"batch={batch_idx + 1}: main_bce={bce_loss.item()}, "
                    f"contrastive={contrastive_loss.item()}, "
                    f"interaction_aux={interaction_aux_loss.item()}, "
                    f"positive_count={int((labels > 0.5).sum().item())}"
                )

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_interaction_aux_loss += interaction_aux_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                attention_model.parameters(), max_norm=5.0,
                error_if_nonfinite=True
            )
            optimizer.step()

            correct += torch.eq(torch.round(logits), labels).sum().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_bce_loss = total_bce_loss / n_batches
        avg_contrastive_loss = total_contrastive_loss / n_batches
        avg_interaction_aux_loss = total_interaction_aux_loss / n_batches
        acc = correct / len(train_loader.dataset)
        effective_interaction_gate = (
            torch.sigmoid(attention_model.interaction_gate_logit).detach().item()
            if attention_model.use_cross_attention else 0.0
        )

        train_losses.append(avg_loss)
        train_bce_losses.append(avg_bce_loss)
        train_contrastive_losses.append(avg_contrastive_loss)
        train_interaction_aux_losses.append(avg_interaction_aux_loss)
        train_accs.append(acc)

        val_results = predicting(attention_model, test_loader, save_path=None, threshold=threshold)
        total_labels = val_results[0] 
        total_preds = val_results[1]  

        metric = calculate_metric(
            (torch.tensor(total_preds) >= threshold).int(),
            torch.tensor(total_labels).int(),
            torch.tensor(total_preds)
        )

        if isinstance(metric, (tuple, list)) and len(metric) >= 8:
            current_mcc = metric[7]
            print(f"Current MCC: {current_mcc}, Best MCC: {best_mcc}")

            if current_mcc > best_mcc:
                best_mcc = current_mcc
                best_epoch = epoch
                best_val_preds = total_preds
                best_val_labels = total_labels

                # Save the model
                model_save_path = os.path.join(result_dir, f"fold_{nn}_best_model.pkl")
                best_model_path = model_save_path
                torch.save({
                    'model_state_dict': attention_model.state_dict(),
                    'epoch': epoch,
                    'mcc': best_mcc,
                    'interaction_aux_loss_coef': interaction_aux_loss_coef,
                    'interaction_gate': effective_interaction_gate,
                    'ablation': train_args['config']['ablation'],
                }, model_save_path)


                save_path = os.path.join(result_dir, f"fold_{nn}_predictions.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                train_fusion_save_path = os.path.join(result_dir, f"fold_{nn}_train_fusion_vectors.npy")
                train_labels_save_path = os.path.join(result_dir, f"fold_{nn}_train_labels.npy")
                predicting(attention_model, train_loader, save_path=None, threshold=threshold,
                        fusion_save_path=train_fusion_save_path, labels_save_path=train_labels_save_path, dataset_type="train")
                test_fusion_save_path = os.path.join(result_dir, f"fold_{nn}_test_fusion_vectors.npy")
                test_labels_save_path = os.path.join(result_dir, f"fold_{nn}_test_labels.npy")
                predicting(attention_model, test_loader, save_path=save_path, threshold=threshold, 
                        fusion_save_path=test_fusion_save_path, labels_save_path=test_labels_save_path, dataset_type="test")

                current_auc = roc_auc_score(total_labels, total_preds)
                current_auprc = average_precision_score(total_labels, total_preds)

                fpr, tpr, thresholds = sklearn.metrics.roc_curve(total_labels, total_preds, pos_label=1)
                best_roc_data = (fpr, tpr, current_auc)

                precision, recall, thresholds = sklearn.metrics.precision_recall_curve(total_labels, total_preds, pos_label=1)
                best_prc_data = (recall, precision, current_auprc)
                                
                metric_mcc = metric
                patience = train_args['config']['train']['patience']
            else:
                patience -= 1
                if patience <= 0:
                    early_stopping_triggered = True
        else:
            print(f"Warning: metric format is incorrect - Type: {type(metric)}, Content: {metric}")
            patience -= 1
            if patience <= 0:
                early_stopping_triggered = True

        elapsed_time = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
        times += elapsed_time

        with open(train_args['rst_file'], 'a+') as fp:
            if isinstance(metric, (tuple, list)) and len(metric) >= 8:
                fp.write(str(nn) + '_fold:' + str(epoch + 1) 
                         + '\ttrainacc=' + str(acc) 
                         + '\ttrainloss=' + str(avg_loss) 
                         + '\ttrain_bce_loss=' + str(avg_bce_loss) 
                         + '\ttrain_contrastive_loss=' + str(avg_contrastive_loss) 
                         + '\ttrain_interaction_aux_loss=' + str(avg_interaction_aux_loss)
                         + '\tzero_positive_batches=' + str(zero_positive_batches)
                         + '\tinteraction_gate=' + str(effective_interaction_gate)
                         + '\tacc=' + str(metric[0].item() if hasattr(metric[0], 'item') else metric[0]) 
                         + '\tprec=' + str(metric[1].item() if hasattr(metric[1], 'item') else metric[1]) 
                         + '\trecall=' + str(metric[2].item() if hasattr(metric[2], 'item') else metric[2]) 
                         + '\tspec=' + str(metric[3].item() if hasattr(metric[3], 'item') else metric[3]) 
                         + '\tf1=' + str(metric[4].item() if hasattr(metric[4], 'item') else metric[4]) 
                         + '\tauc=' + str(metric[5].item() if hasattr(metric[5], 'item') else metric[5]) 
                         + '\tauprc=' + str(metric[6].item() if hasattr(metric[6], 'item') else metric[6]) 
                         + '\tmcc=' + str(metric[7].item() if hasattr(metric[7], 'item') else metric[7]) 
                         + '\ttime=' + str(elapsed_time) 
                         + '\n')
            else:
                fp.write(str(nn) + '_fold:' + str(epoch + 1) 
                         + '\ttrainacc=' + str(acc) 
                         + '\ttrainloss=' + str(avg_loss) 
                         + '\ttrain_bce_loss=' + str(avg_bce_loss) 
                         + '\ttrain_contrastive_loss=' + str(avg_contrastive_loss) 
                         + '\ttrain_interaction_aux_loss=' + str(avg_interaction_aux_loss)
                         + '\tzero_positive_batches=' + str(zero_positive_batches)
                         + '\tinteraction_gate=' + str(effective_interaction_gate)
                         + '\tmetric_error=True'
                         + '\ttime=' + str(elapsed_time) 
                         + '\n')
    
    if best_model_path is None:
        raise RuntimeError(f"Fold {nn} completed without a saved checkpoint")
    if attention_model.use_cross_attention:
        export_test_pair_attention_analysis(
            best_model_path, test_loader, result_dir, nn,
            config=train_args['config'], threshold=threshold
        )
    else:
        with open(
            os.path.join(result_dir, "ATTENTION_DISABLED.txt"),
            "w", encoding="utf-8"
        ) as handle:
            handle.write(
                "Cross-attention is disabled for this ablation; no residue "
                "scores or attention matrices are defined.\n"
            )
    return metric_mcc, best_epoch, times

def export_test_pair_attention_analysis(model_path, test_loader, result_dir,
                                        fold_idx, config, threshold=0.5):
    """Export bidirectional attention outputs for every outer-test pair."""
    print(f"Exporting all test-pair attention results for fold {fold_idx}...")
    device = get_device()
    model = build_model(config).to(device)
    
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load fold {fold_idx} checkpoint {model_path}"
        ) from e
    
    results = predicting(model, test_loader, collect_attention=True)
    (total_labels, total_preds, residue_scores1, residue_scores2, 
     seqs1, seqs2, attention_weights, total_protein_ids1, total_protein_ids2) = results
    
    analysis_dir = os.path.join(result_dir, "test_pair_attention_analysis")
    os.makedirs(analysis_dir, exist_ok=False)
    manifest_path = os.path.join(analysis_dir, "manifest.tsv")

    expected_count = len(total_preds)
    output_lengths = {
        'labels': len(total_labels),
        'protein1_ids': len(total_protein_ids1),
        'protein2_ids': len(total_protein_ids2),
        'protein1_sequences': len(seqs1),
        'protein2_sequences': len(seqs2),
        'protein1_residue_scores': len(residue_scores1),
        'protein2_residue_scores': len(residue_scores2),
        'forward_attention': len(attention_weights['cross_attn']),
        'reverse_attention': len(attention_weights['cross_attn_reverse']),
    }
    mismatched = {
        name: length for name, length in output_lengths.items()
        if length != expected_count
    }
    if mismatched:
        raise RuntimeError(
            f"Incomplete fold {fold_idx} attention export: expected "
            f"{expected_count} samples, got {mismatched}"
        )

    with open(manifest_path, 'w', encoding='utf-8') as manifest:
        manifest.write(
            "Fold\tTest_Index\tProtein1_ID\tProtein2_ID\tTrue_Label\t"
            "Predicted_Probability\tPredicted_Label\tPair_Directory\n"
        )
        for idx in range(len(total_preds)):
            if idx >= len(attention_weights['cross_attn']):
                raise IndexError(
                    f"Test index {idx} has no forward attention matrix"
                )
            if idx >= len(attention_weights['cross_attn_reverse']):
                raise IndexError(
                    f"Test index {idx} has no reverse attention matrix"
                )

            protein1_id = total_protein_ids1[idx]
            protein2_id = total_protein_ids2[idx]
        
            pair_directory = (
                f"{idx + 1:04d}__{protein1_id}__{protein2_id}"
            )
            sample_dir = os.path.join(analysis_dir, pair_directory)
            os.makedirs(sample_dir, exist_ok=False)

            sample_seq1 = seqs1[idx]
            sample_seq2 = seqs2[idx]
            sample_score1 = residue_scores1[idx]
            sample_score2 = residue_scores2[idx]
            forward_attention = attention_weights['cross_attn'][idx]
            reverse_attention = attention_weights['cross_attn_reverse'][idx]
        
            key_residues1 = analyze_key_residues(
                sequence=sample_seq1,
                residue_scores=sample_score1,
                save_path=os.path.join(
                    sample_dir, f"{protein1_id}_key_residues.svg"
                )
            )
        
            key_residues2 = analyze_key_residues(
                sequence=sample_seq2,
                residue_scores=sample_score2,
                save_path=os.path.join(
                    sample_dir, f"{protein2_id}_key_residues.svg"
                )
            )
        
            save_key_residues_report(
                key_residues1, sample_seq1,
                os.path.join(sample_dir, f"{protein1_id}_key_residues.txt"),
                protein_name=(
                    f"{protein1_id} (from pair: "
                    f"{protein1_id}_{protein2_id})"
                )
            )
        
            save_key_residues_report(
                key_residues2, sample_seq2,
                os.path.join(sample_dir, f"{protein2_id}_key_residues.txt"),
                protein_name=(
                    f"{protein2_id} (from pair: "
                    f"{protein1_id}_{protein2_id})"
                )
            )

            for protein_id, sequence, scores in (
                (protein1_id, sample_seq1, sample_score1),
                (protein2_id, sample_seq2, sample_score2),
            ):
                score_path = os.path.join(
                    sample_dir, f"{protein_id}_all_residue_scores.tsv"
                )
                with open(score_path, 'w', encoding='utf-8') as score_file:
                    score_file.write(
                        "Position\tAmino_Acid\tAttention_Derived_Score\n"
                    )
                    for position, (amino_acid, score) in enumerate(
                        zip(sequence, scores), start=1
                    ):
                        score_file.write(
                            f"{position}\t{amino_acid}\t{float(score):.8g}\n"
                        )
        
            visualize_attention_heatmap(
                sequence1=sample_seq1,
                sequence2=sample_seq2,
                attention_weights=forward_attention,
                save_path=os.path.join(
                    sample_dir,
                    f"{protein1_id}_to_{protein2_id}_attention.svg"
                ),
                raw_data_path=os.path.join(
                    sample_dir,
                    f"{protein1_id}_to_{protein2_id}_attention.csv"
                ),
            )
            visualize_attention_heatmap(
                sequence1=sample_seq2,
                sequence2=sample_seq1,
                attention_weights=reverse_attention,
                save_path=os.path.join(
                    sample_dir,
                    f"{protein2_id}_to_{protein1_id}_attention.svg"
                ),
                raw_data_path=os.path.join(
                    sample_dir,
                    f"{protein2_id}_to_{protein1_id}_attention.csv"
                ),
            )

            probability = float(total_preds[idx])
            true_label = int(total_labels[idx])
            predicted_label = int(probability >= threshold)
            manifest.write(
                f"{fold_idx}\t{idx + 1}\t{protein1_id}\t{protein2_id}\t"
                f"{true_label}\t{probability:.8g}\t{predicted_label}\t"
                f"{pair_directory}\n"
            )

    print(
        f"Saved attention analyses for all {len(total_preds)} test pairs to "
        f"{analysis_dir}"
    )


def save_fold_data(save_dir, train_pairs, test_pairs):
    """Save each fold assignment to the requested directory"""
    # Ensure that the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving fold data to: {save_dir}")
    
    # Save training and test pairs as NumPy arrays
    np.save(os.path.join(save_dir, 'train_pairs.npy'), train_pairs)
    np.save(os.path.join(save_dir, 'test_pairs.npy'), test_pairs)
    
    # Save training and test pairs as tab-separated text
    with open(os.path.join(save_dir, 'train_pairs.txt'), 'w') as f:
        for pair, label in train_pairs:
            p1, p2 = pair.split('_')
            f.write(f"{p1}\t{p2}\t{label}\n")
            
    with open(os.path.join(save_dir, 'test_pairs.txt'), 'w') as f:
        for pair, label in test_pairs:
            p1, p2 = pair.split('_')
            f.write(f"{p1}\t{p2}\t{label}\n")


def main(config_path):
    config = get_config(config_path)
    settings = resolve_ablation_config(config)
    set_seed(config['train']['seed'])
    device = get_device()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_result_dir = config['output']['model_save_dir']
    if not os.path.isabs(base_result_dir):
        base_result_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(config_path)), base_result_dir
        ))
        config['output']['model_save_dir'] = base_result_dir
    result_dir = os.path.join(base_result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Ablation experiment: {settings['name']}")
    print(f"Ablation switches: {settings}")
    threshold = config['train'].get('threshold', 0.5)  
    config['output']['rst_file'] = os.path.join(result_dir, "results.tsv")
    source_archive_dir = save_source_code(
        result_dir=result_dir,
        config_path=config_path,
        resolved_config=config,
        source_dir=os.path.dirname(os.path.abspath(__file__))
    )
    print(f"Source code snapshot saved to: {source_archive_dir}")
    split_seed = config['train'].get('split_seed', config['train']['seed'])
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=split_seed
    )
    X, y = Dataset.load_pair(config['data']['pair_path'])
    metrics = {'mcc': 0., 'acc': 0., 'auc': 0., 'prec': 0., 
              'spec': 0., 'recall': 0., 'f1': 0., 'auprc': 0., 'times': 0}
    with open(config['output']['rst_file'], 'a+') as f:
        f.write(f"Experiment started at: {timestamp}\n\n")
        for section, params in config.items():
            f.write(f"\n[{section}]\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        fold_result_dir = os.path.join(result_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_result_dir, exist_ok=True)
        train_pairs = [[X[i], y[i]] for i in train_index]
        test_pairs = [[X[i], y[i]] for i in test_index]

        fold_dir = os.path.join(result_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        save_fold_data(fold_dir, train_pairs, test_pairs)
        train_args = {
            'epochs': config['train']['epoch'],
            'lr': config['train']['lr'],
            'model': build_model(config).to(device),
            'rst_file': config['output']['rst_file'],
            'contrastive_loss_coef': config['train']['contrastive_loss_coef'],
            'interaction_aux_loss_coef': config['train']['interaction_aux_loss_coef'],
            'config': config
        }
        train_args['optimizer'] = torch.optim.AdamW(
            train_args['model'].parameters(), 
            lr=train_args['lr']
        )
        train_args['criterion'] = torch.nn.BCELoss()
        train_pairs, train_labels = zip(*train_pairs)
        test_pairs, test_labels = zip(*test_pairs)

        train_dataset = Dataset(train_pairs, train_labels, config)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config['train']['batchsize'],
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn  
        )
        train_args['train_loader'] = train_loader

        test_dataset = Dataset(test_pairs, test_labels, config)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config['train']['batchsize'],
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn  
        )

        metric_mcc_one, best_epoch, times = train(
            train_args, 
            fold_idx + 1, 
            test_loader,
            fold_result_dir,
            threshold=threshold 
        )
            
        metrics['mcc'] += metric_mcc_one[7]     
        metrics['acc'] += metric_mcc_one[0]      
        metrics['auc'] += metric_mcc_one[5]      
        metrics['prec'] += metric_mcc_one[1]     
        metrics['spec'] += metric_mcc_one[3]     
        metrics['recall'] += metric_mcc_one[2]  
        metrics['f1'] += metric_mcc_one[4]       
        metrics['auprc'] += metric_mcc_one[6]    
        metrics['times'] += times

    with open(config['output']['rst_file'], 'a+') as f:
        f.write('\n\nFinal Results:\n')

        ordered_metrics = ['mcc', 'acc', 'auc', 'prec', 'spec', 'recall', 'f1', 'auprc']
        for key in ordered_metrics:
            f.write(f'{key}_mean: {"%.6f" % (metrics[key] / 5)}\n')
            
        f.write(f'Total time: {metrics["times"] // 3600}h {(metrics["times"] % 3600) // 60}m {metrics["times"] % 60}s\n')

        total_params = sum(p.numel() for p in train_args['model'].parameters())
        f.write(f'Total model parameters: {total_params}\n')

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            f.write(f'GPU memory allocated: {mem_allocated:.3f} GB\n')
            f.write(f'GPU memory reserved: {mem_reserved:.3f} GB\n')
        else:
            f.write('GPU not available, memory usage not recorded.\n')

    return result_dir

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_model.yml")
    args = parser.parse_args()
    result_dir = main(args.config)
    print(f"Training complete! Results saved to: {result_dir}")
