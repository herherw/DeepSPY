import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import yaml
import os
import hashlib
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sklearn
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc


def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively merge a configuration override into a base dictionary."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path: str) -> Dict:
    """Load YAML, resolve optional inheritance, and normalize file paths."""
    config_file = Path(config_path).expanduser().resolve()
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    parent = config.pop('extends', None)
    if parent:
        parent_path = (config_file.parent / parent).resolve()
        config = _deep_update(load_config(str(parent_path)), config)

    project_root_setting = config.get('project_root')
    if project_root_setting:
        project_root = (config_file.parent / project_root_setting).resolve()
        config['project_root'] = str(project_root)
        for section_name in ('data', 'output'):
            section = config.get(section_name, {})
            for key, value in list(section.items()):
                if key.endswith(('_path', '_dir')) and isinstance(value, str):
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        section[key] = str((project_root / path).resolve())
    return config

def get_config(config_path: Optional[str] = None) -> Dict:
    """Get configuration with command line parameter override support"""
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[1] / 'configs' / 'benchmark_450ep_t050.yml'
        )
    
    return load_config(config_path)


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_source_code(result_dir: str, config_path: Optional[str] = None,
                     resolved_config: Optional[Dict] = None,
                     source_dir: Optional[str] = None) -> str:
    """Archive exact runtime source, resolved config, and environment metadata."""
    source_root = Path(
        source_dir or Path(__file__).resolve().parent
    ).resolve()
    archive_dir = Path(result_dir).resolve() / 'source_code'
    archive_dir.mkdir(parents=True, exist_ok=False)

    copied_files = []
    source_files = sorted(source_root.glob('*.py'), key=lambda path: path.name)
    if not source_files:
        raise FileNotFoundError(f"No Python source files found in {source_root}")
    for source_path in source_files:
        destination = archive_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(destination)

    if config_path is not None:
        config_source = Path(config_path).expanduser().resolve()
        if not config_source.is_file():
            raise FileNotFoundError(f"Training config not found: {config_source}")
        config_destination = archive_dir / config_source.name
        shutil.copy2(config_source, config_destination)
        copied_files.append(config_destination)

    if resolved_config is not None:
        resolved_path = archive_dir / 'resolved_config.yml'
        with open(resolved_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(
                resolved_config, handle, sort_keys=False, allow_unicode=True
            )
        copied_files.append(resolved_path)

    runtime_metadata = {
        'archived_at': datetime.now().astimezone().isoformat(),
        'command': sys.argv,
        'working_directory': os.getcwd(),
        'source_directory': str(source_root),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'scikit_learn_version': sklearn.__version__,
        'pyyaml_version': getattr(yaml, '__version__', None),
        'torch_version': str(torch.__version__),
        'cuda_runtime_version': (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        'cudnn_version': (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None
        ),
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        runtime_metadata['cuda_device'] = torch.cuda.get_device_name(0)

    metadata_path = archive_dir / 'runtime_environment.yml'
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(
            runtime_metadata, handle, sort_keys=False, allow_unicode=True
        )
    copied_files.append(metadata_path)

    manifest_path = archive_dir / 'SHA256SUMS.txt'
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        for copied_path in sorted(copied_files, key=lambda path: path.name):
            handle.write(
                f"{_sha256_file(copied_path)}  {copied_path.name}\n"
            )

    return str(archive_dir)

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
    
    if scores.ndim == 1:  # One-dimensional input.
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
    
    if isinstance(residue_scores, torch.Tensor):
        residue_scores = residue_scores.detach().cpu().numpy()
    residue_scores = np.asarray(residue_scores).squeeze()
    if residue_scores.ndim != 1 or len(residue_scores) != len(sequence):
        raise ValueError(
            "Residue scores must map one-to-one to the biological sequence: "
            f"scores shape={residue_scores.shape}, sequence length={len(sequence)}"
        )
    
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
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = np.asarray(attention_weights)
    expected_shape = (seq1_len, seq2_len)
    if attention_weights.shape != expected_shape:
        raise ValueError(
            "Attention matrix must map one-to-one to both biological sequences: "
            f"matrix shape={attention_weights.shape}, expected={expected_shape}"
        )
    
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
