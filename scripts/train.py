#!/usr/bin/env python
# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import sys
import os
import time
import shutil
from tqdm import tqdm
from model import Model, sup_contrasitive_loss, attention_consistency_loss
from data_loader import Dataset, collate_fn
from util import get_config, calculate_metric, analyze_key_residues, visualize_attention_heatmap, save_key_residues_report
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

def predicting(model, loader, save_path=None, threshold=0.5, fusion_save_path=None, labels_save_path=None, dataset_type="test"):
    model.eval()
    device = get_device()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    total_residue_scores1 = []
    total_residue_scores2 = []
    total_cross_attn = []
    total_seqs1 = []
    total_seqs2 = []
    total_fusion_vectors = [] 
    total_protein_ids1 = []
    total_protein_ids2 = []

    print(f'Make prediction for {dataset_type} dataset with {len(loader.dataset)} samples...')
    with torch.no_grad():
        for _, (ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, y) in enumerate(tqdm(loader)):
            batch_data = [ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, y]
            batch_data = tuple(data.to(device) if isinstance(data, torch.Tensor) else data for data in batch_data)
            ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths, name1, name2, y = batch_data

            out = model(ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths)
            logits = out['logits'].squeeze(-1)
            total_preds = torch.cat((total_preds, logits), 0)
            total_labels = torch.cat((total_labels, y.float()), 0)
            residue_scores1, residue_scores2 = out['residue_scores']
            attn_weights = out['attention_weights']
            fusion_vector = out['fusion_vector']  
            total_fusion_vectors.append(fusion_vector)  
            for i in range(len(name1)):
                total_protein_ids1.append(name1[i])
                total_protein_ids2.append(name2[i])
                seq1 = loader.dataset.get_protein_sequence(name1[i], is_first_protein=True)
                seq2 = loader.dataset.get_protein_sequence(name2[i], is_first_protein=False)
                total_seqs1.append(seq1)
                total_seqs2.append(seq2)
                total_residue_scores1.append(residue_scores1[i].cpu().numpy())
                total_residue_scores2.append(residue_scores2[i].cpu().numpy())

            total_cross_attn.append(attn_weights['cross_attn'].cpu().numpy())

    total_cross_attn = [attn for batch in total_cross_attn for attn in batch]


    if fusion_save_path:  # Check if fusion_save_path is None
        total_fusion_vectors = np.concatenate(total_fusion_vectors, axis=0)  
        fusion_file = fusion_save_path.replace("{dataset_type}", dataset_type)
        np.save(fusion_file, total_fusion_vectors)
    
    if labels_save_path:  # Check if labels_save_path is None
        labels_file = labels_save_path.replace("{dataset_type}", dataset_type)
        np.save(labels_file, total_labels.cpu().numpy())
        
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Protein1_ID\tProtein2_ID\tPredicted_Probability\tBinary_Label\n")
            for p1, p2, prob in zip(total_protein_ids1, total_protein_ids2, total_preds.cpu().numpy().flatten()):
                label = 1.0 if prob >= threshold else 0.0
                f.write(f"{p1}\t{p2}\t{prob:.6f}\t{label}\n")

    return (total_labels.cpu().numpy().flatten(), 
            total_preds.cpu().numpy().flatten(),
            total_residue_scores1,
            total_residue_scores2,
            total_seqs1,
            total_seqs2,
            {
                'cross_attn': total_cross_attn
            },
            total_protein_ids1,
            total_protein_ids2
    )

def train(train_args, nn, test_loader, result_dir, threshold=0.5):
    """Train the model"""
    train_losses = []
    train_bce_losses = []
    train_contrastive_losses = []
    train_consistency_losses = []
    train_accs = []
    times = 0.0
    patience = train_args['config']['train']['patience']  
    early_stopping_triggered = False
    attention_model = train_args['model']
    optimizer = train_args['optimizer']
    criterion = train_args["criterion"]
    contrastive_loss_coef = train_args["contrastive_loss_coef"]
    consistency_loss_coef = train_args["consistency_loss_coef"]  
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
        total_consistency_loss = 0
        correct = 0
        n_batches = 0
        attention_model.train()

        for _, (ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, 
                        esmc1_lengths, esmc2_lengths, name1, name2, labels) in enumerate(tqdm(train_loader)):

            batch_data = [ckesm1, ckesm2, aac1, aac2, dip1, dip2, 
                        esmc1, esmc2, esmc1_lengths, esmc2_lengths, labels]
            batch_data = tuple(data.to(device) if isinstance(data, torch.Tensor) else data for data in batch_data)
            ckesm1, ckesm2, aac1, aac2, dip1, dip2, esmc1, esmc2, esmc1_lengths, esmc2_lengths, labels = batch_data
            
            optimizer.zero_grad()
            
            outputs = attention_model(ckesm1, ckesm2, aac1, aac2, dip1, dip2, 
                                    esmc1, esmc2, esmc1_lengths, esmc2_lengths)
            cross_attn = outputs['attention_weights']['cross_attn']
            logits = outputs['logits'].squeeze(-1)
            seq1_projections, seq2_projections = outputs['projection']
            labels = labels.float()

            bce_loss = criterion(logits, labels)
            consistency_loss = attention_consistency_loss(cross_attn, weight_factor=2.0)
            contrastive_loss = sup_contrasitive_loss(seq1_projections, seq2_projections, labels)
            loss = bce_loss + contrastive_loss_coef * contrastive_loss + consistency_loss_coef * consistency_loss

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_consistency_loss += consistency_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), max_norm=5.0) 
            optimizer.step()

            correct += torch.eq(torch.round(logits), labels).sum().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_bce_loss = total_bce_loss / n_batches
        avg_contrastive_loss = total_contrastive_loss / n_batches
        avg_consistency_loss = total_consistency_loss / n_batches
        acc = correct / len(train_loader.dataset)

        train_losses.append(avg_loss)
        train_bce_losses.append(avg_bce_loss)
        train_contrastive_losses.append(avg_contrastive_loss)
        train_consistency_losses.append(avg_consistency_loss)
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

            if current_mcc >= best_mcc:
                best_mcc = current_mcc
                best_epoch = epoch
                best_val_preds = total_preds
                best_val_labels = total_labels

                # Save model
                model_save_path = os.path.join(result_dir, f"fold_{nn}_best_model.pkl")
                torch.save({
                    'model_state_dict': attention_model.state_dict(),
                    'epoch': epoch,
                    'mcc': best_mcc
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

                evaluate_key_residues(model_save_path, test_loader, result_dir, nn)
                
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
                         + '\ttrain_consistency_loss=' + str(avg_consistency_loss) 
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
                         + '\ttrain_consistency_loss=' + str(avg_consistency_loss) 
                         + '\tmetric_error=True'
                         + '\ttime=' + str(elapsed_time) 
                         + '\n')
    
    return metric_mcc, best_epoch, times

def evaluate_key_residues(model_path, test_loader, result_dir, fold_idx, threshold=0.5):
    print(f"Evaluating key residues for fold {fold_idx}...")
    config = get_config("./config.yml")
    device = get_device()
    model = Model(
        embed_hid_dim=config['model']['embed_hid_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    results = predicting(model, test_loader)
    (total_labels, total_preds, residue_scores1, residue_scores2, 
     seqs1, seqs2, attention_weights, total_protein_ids1, total_protein_ids2) = results
    
    eval_dir = os.path.join(result_dir, "evaluation")
    if os.path.exists(eval_dir):
        try:
            shutil.rmtree(eval_dir)
        except Exception as e:
            print(f"Failed to remove evaluation directory: {e}")
        os.makedirs(eval_dir, exist_ok=True)
    

    positive_indices = np.where(total_preds >= threshold)[0][:5]
    negative_indices = np.where(total_preds < threshold)[0][:5]    
    selected_indices = np.concatenate([positive_indices, negative_indices])
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(attention_weights['cross_attn']):
            print(f"Warning: Index {idx} is out of range for attention_weights['cross_attn']")
            continue

        protein1_id = total_protein_ids1[idx]
        protein2_id = total_protein_ids2[idx]
        
        sample_dir = os.path.join(eval_dir, f"{protein1_id}_{protein2_id}")
        os.makedirs(sample_dir, exist_ok=True)

        sample_seq1 = seqs1[idx]
        sample_seq2 = seqs2[idx]
        sample_score1 = residue_scores1[idx]
        sample_score2 = residue_scores2[idx]
        sample_attention_weights = attention_weights['cross_attn'][idx]
        
        key_residues1 = analyze_key_residues(
            sequence=sample_seq1,
            residue_scores=sample_score1,
            save_path=os.path.join(sample_dir, f"{protein1_id}_key_residues.svg")
        )
        
        key_residues2 = analyze_key_residues(
            sequence=sample_seq2,
            residue_scores=sample_score2,
            save_path=os.path.join(sample_dir, f"{protein2_id}_key_residues.svg")
        )
        
        save_key_residues_report(
            key_residues1, sample_seq1,
            os.path.join(sample_dir, f"{protein1_id}_key_residues.txt"),
            protein_name=f"{protein1_id} (from pair: {protein1_id}_{protein2_id})"
        )
        
        save_key_residues_report(
            key_residues2, sample_seq2,
            os.path.join(sample_dir, f"{protein2_id}_key_residues.txt"),
            protein_name=f"{protein2_id} (from pair: {protein1_id}_{protein2_id})"
        )
        
        visualize_attention_heatmap(
            sequence1=sample_seq1,
            sequence2=sample_seq2,
            attention_weights=sample_attention_weights,
            save_path=os.path.join(sample_dir, f"{protein1_id}_{protein2_id}_attention_heatmap.svg"),
            raw_data_path=os.path.join(sample_dir, f"{protein1_id}_{protein2_id}_attention_weights.csv"),
        )


def save_fold_data(save_dir, train_pairs, test_pairs):
    """Save each fold's data to the specified directory"""
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving fold data to: {save_dir}")
    
    # Save training and testing pairs as npy files
    np.save(os.path.join(save_dir, 'train_pairs.npy'), train_pairs)
    np.save(os.path.join(save_dir, 'test_pairs.npy'), test_pairs)
    
    # Save training and testing pairs as txt files
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
    set_seed(config['train']['seed'])
    device = get_device()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_result_dir = config['output']['model_save_dir']
    result_dir = os.path.join(base_result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    threshold = config['train'].get('threshold', 0.5)  
    config['output']['rst_file'] = os.path.join(result_dir, "results.tsv")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['train']['seed'])
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
            'model': Model(
                            embed_hid_dim=config['model']['embed_hid_dim'],
                            dropout=config['model']['dropout']
                            ).to(device),
            'rst_file': config['output']['rst_file'],
            'contrastive_loss_coef': config['train']['contrastive_loss_coef'],
            'consistency_loss_coef': config['train']['consistency_loss_coef'],
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
    config_path = "./config.yml"  
    result_dir = main(config_path)
    print(f"Training complete! Results saved to: {result_dir}")
