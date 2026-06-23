import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import yaml
import os
from typing import Dict, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_config(config_path: Optional[str] = None) -> Dict:
    """Get configuration with command line parameter override support"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), './config.yml')
    
    return load_config(config_path)

def calculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    ACC = float(tp + tn) / test_num

    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels = labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()

    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    AUCPR = auc(recall, precision)
    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, AUCPR, MCC, tp, fp, tn, fn])


    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return (ACC, Precision, Recall, Specificity, F1, AUC, AP, MCC)

def adapt_scores_to_sequence(scores, sequence_len):
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.array(scores)
    
    if scores.ndim == 1:  
        scores_len = len(scores)
        if scores_len == sequence_len:
            return scores
        elif scores_len < sequence_len:
            indices = np.linspace(0, scores_len - 1, sequence_len)
            return np.interp(indices, np.arange(scores_len), scores)
        else:
            return scores[:sequence_len]
    elif scores.ndim == 2:  
        scores_len, other_dim = scores.shape
        if scores_len == sequence_len:
            return scores
        elif scores_len < sequence_len:
            indices = np.linspace(0, scores_len - 1, sequence_len)
            expanded_scores = np.zeros((sequence_len, other_dim))
            for i in range(other_dim):
                expanded_scores[:, i] = np.interp(indices, np.arange(scores_len), scores[:, i])
            return expanded_scores
        else:
            return scores[:sequence_len, :]
    else:
        raise ValueError("Unsupported scores dimension")

def analyze_key_residues(sequence, residue_scores, save_path=None, 
                                        threshold_percentile=90, figsize=(12, 4)):
    if isinstance(sequence, (list, tuple)):
        sequence = str(sequence[0]) if len(sequence) > 0 else ""
    
    residue_scores = adapt_scores_to_sequence(residue_scores, len(sequence))
    
    threshold = np.percentile(residue_scores, threshold_percentile)
    
    key_residues = []
    for i, score in enumerate(residue_scores):
        if score >= threshold:
            key_residues.append((i + 1, sequence[i], float(score)))  
    
    if save_path:
        plt.figure(figsize=figsize)
        plt.bar(range(1, len(residue_scores) + 1), residue_scores, color='lightgray', alpha=0.7)
        key_positions = [res[0] for res in key_residues]
        key_scores = [res[2] for res in key_residues]
        plt.bar(key_positions, key_scores, color='red', alpha=0.7)
        for pos, score in zip(key_positions, key_scores):
            plt.text(pos, score, f"{sequence[pos - 1]}", ha='center', va='bottom', fontsize=10)
        plt.xlabel("Residue Position", fontsize=16)
        plt.ylabel("Attention Score", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, len(sequence) + 1)
        plt.ylim(0, max(residue_scores) * 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, format='svg', bbox_inches='tight')
        plt.close()
    
    return key_residues

def visualize_attention_heatmap(sequence1, sequence2, attention_weights, save_path, raw_data_path=None,
                                figsize=(10, 8), cmap="YlOrRd", fontsize=12):
    sequence1 = str(sequence1)
    sequence2 = str(sequence2)
    
    seq1_len, seq2_len = len(sequence1), len(sequence2)
    attention_weights = adapt_scores_to_sequence(attention_weights, seq1_len)
    attention_weights = adapt_scores_to_sequence(attention_weights.T, seq2_len).T
    
    if raw_data_path:
        attention_df = pd.DataFrame(attention_weights, index=list(sequence1), columns=list(sequence2))
        attention_df.to_csv(raw_data_path, index=True, header=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attention_weights,
        cmap=cmap,
        cbar=True,
        cbar_kws={"shrink": 0.5, "pad": 0.02},
        ax=ax
        # cbar_kws={"label": "Attention Score"}
    )
    cbar = ax.collections[0].colorbar   
    cbar.ax.tick_params(labelsize=fontsize)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_key_residues_report(key_residues, sequence, save_path, protein_name=""):
    with open(save_path, 'w') as f:
        f.write(f"Key Residues Report for Protein {protein_name}: {sequence}\n")
        f.write("="*60 + "\n\n")
        f.write("Rank\tPosition\tAmino Acid\tImportance Score\n")
        f.write("-"*60 + "\n")
        
        for i, (pos, aa, score) in enumerate(key_residues):
            f.write(f"{i+1}\t{pos}\t{aa}\t{score:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Total key residues identified: {len(key_residues)}\n")

