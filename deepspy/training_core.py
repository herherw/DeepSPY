"""Shared DeepSPY training and evaluation, based on the 20261104_195520 snapshot."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from torch.utils.data import DataLoader

from .data import Dataset, collate_fn
from .model import Model, pairwise_contrastive_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(pairs, labels, config, shuffle=False):
    dataset = Dataset(pairs, labels, config)
    # The projector contains BatchNorm1d; avoid a final singleton batch.
    batch_size = int(config['train']['batchsize'])
    if shuffle and len(dataset) % batch_size == 1 and len(dataset) > batch_size:
        from torch.utils.data import Sampler
        class NoSingletonBatches(Sampler):
            def __iter__(self):
                order = torch.randperm(len(dataset)).tolist()
                batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
                batches[-2].extend(batches.pop())
                yield from batches
            def __len__(self):
                return (len(dataset) + batch_size - 1) // batch_size - 1
        return DataLoader(dataset, batch_sampler=NoSingletonBatches(), collate_fn=collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, collate_fn=collate_fn)


def make_model(config, device):
    return Model(embed_hid_dim=config['model']['embed_hid_dim'],
                 dropout=config['model']['dropout'],
                 **config.get('ablation', {})).to(device)


def train_epoch(model, loader, optimizer, config, device):
    model.train()
    totals = dict(loss=0., bce=0., contrastive=0., auxiliary=0.)
    count = 0
    settings = config['train']
    for batch in loader:
        inputs = [x.to(device) for x in batch[:10]]
        labels = batch[-1].to(device).float()
        optimizer.zero_grad()
        output = model(*inputs)
        logits = output['logits'].reshape(-1)
        raw_logits = output['raw_logits'].reshape(-1)
        if not torch.isfinite(raw_logits).all():
            raise FloatingPointError('Nonfinite logits')
        bce = F.binary_cross_entropy(logits, labels)
        auxiliary = bce.new_zeros(())
        if model.use_cross_attention and float(settings['interaction_aux_loss_coef']):
            auxiliary_logits = output['interaction_raw_logits'].reshape(-1)
            if not torch.isfinite(auxiliary_logits).all():
                raise FloatingPointError('Nonfinite auxiliary logits')
            auxiliary = F.binary_cross_entropy_with_logits(auxiliary_logits, labels)
        contrastive = bce.new_zeros(())
        if model.use_contrastive_loss:
            first, second = output['projection']
            contrastive = pairwise_contrastive_loss(
                first, second, labels,
                margin=float(settings['contrastive_margin']),
                positive_weight=float(settings['contrastive_positive_weight']),
                negative_weight=float(settings['contrastive_negative_weight']))
        loss = bce + float(settings['contrastive_loss_coef']) * contrastive + float(settings['interaction_aux_loss_coef']) * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        for key, value in [('loss', loss), ('bce', bce), ('contrastive', contrastive), ('auxiliary', auxiliary)]:
            totals[key] += float(value.detach()) * len(labels)
        count += len(labels)
    return {key: value / count for key, value in totals.items()}


def scores(model, loader, device):
    model.eval()
    labels, values = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(*[x.to(device) for x in batch[:10]])
            probability = output['logits'].reshape(-1)
            if not torch.isfinite(probability).all():
                raise FloatingPointError('Nonfinite prediction')
            labels.extend(batch[-1].int().tolist())
            values.extend(probability.cpu().tolist())
    return labels, values


def metrics(labels, values, threshold):
    prediction = [int(x >= threshold) for x in values]
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    result = dict(n=len(labels), positive=sum(labels), threshold=float(threshold),
                  tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                  accuracy=accuracy_score(labels, prediction),
                  precision=precision_score(labels, prediction, zero_division=0),
                  recall=recall_score(labels, prediction, zero_division=0),
                  f1=f1_score(labels, prediction, zero_division=0),
                  mcc=matthews_corrcoef(labels, prediction))
    result['specificity'] = tn / (tn + fp) if tn + fp else 0.
    result['auroc'] = roc_auc_score(labels, values) if len(set(labels)) == 2 else float('nan')
    result['ap'] = average_precision_score(labels, values) if sum(labels) else float('nan')
    return result


def write_tsv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(fold, pairs, labels, values, threshold, epoch):
    return [dict(Fold=fold, Protein1_ID=pair.split('_', 1)[0],
                 Protein2_ID=pair.split('_', 1)[1], Predicted_Probability=value,
                 True_Label=label, Predicted_Label=int(value >= threshold),
                 Threshold=threshold, Selected_Epoch=epoch)
            for pair, label, value in zip(pairs, labels, values)]


def save_pair_split(directory, name, pairs, labels):
    with open(directory / f'{name}_pairs.txt', 'w', encoding='utf-8') as handle:
        for pair, label in zip(pairs, labels):
            handle.write(f'{pair}\t{label}\n')


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=float(config['train']['lr']))

