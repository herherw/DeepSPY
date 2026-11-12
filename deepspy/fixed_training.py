"""Train five final-epoch models once, then score every configured threshold."""

import numpy as np
import torch

from .training_core import (make_loader, make_model,
                           metrics, optimizer_for, prediction_rows,
                           save_pair_split, scores, seed_all, train_epoch,
                           write_tsv)

METRIC_NAMES = ['accuracy', 'precision', 'recall', 'specificity',
                'f1', 'auroc', 'ap', 'mcc']


def threshold_tag(value):
    precise = f"{value:.10f}".rstrip('0').rstrip('.')
    return f"threshold_{precise}".replace('.', 'p')


def validate_thresholds(values):
    thresholds = [round(float(value), 10) for value in values]
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError('threshold_candidates must be inside (0, 1)')
    if len(set(thresholds)) != len(thresholds):
        raise ValueError('threshold_candidates contains duplicates')
    return thresholds


def train_once(config, pairs, labels, splits, root, device):
    """Return five held-out prediction sets in fold order."""
    settings = config['train']
    epochs = int(settings['epoch'])
    shared = root / 'shared_training'
    shared.mkdir()
    seed_all(int(settings['seed']))
    fold_results = []
    shared_predictions = []
    for fold, (fit_idx, test_idx) in enumerate(splits, 1):
        fold_dir = shared / f'fold_{fold}'
        fold_dir.mkdir()
        fit_pairs = [pairs[index] for index in fit_idx]
        fit_labels = [labels[index] for index in fit_idx]
        test_pairs = [pairs[index] for index in test_idx]
        test_labels = [labels[index] for index in test_idx]
        save_pair_split(fold_dir, 'train', fit_pairs, fit_labels)
        save_pair_split(fold_dir, 'test', test_pairs, test_labels)
        train_loader = make_loader(fit_pairs, fit_labels, config, shuffle=True)
        test_loader = make_loader(test_pairs, test_labels, config)
        model = make_model(config, device)
        optimizer = optimizer_for(model, config)
        history = []
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, config, device)
            history.append(dict(fold=fold, epoch=epoch, **loss))
            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                print(f'fold={fold} epoch={epoch}/{epochs} '
                      f'loss={loss["loss"]:.5f}', flush=True)
        write_tsv(fold_dir / 'training_history.tsv', history)
        checkpoint = fold_dir / f'fold_{fold}_epoch_{epochs}_model.pkl'
        torch.save(dict(model_state_dict=model.state_dict(), epoch=epochs,
                        threshold=float(settings['threshold']), config=config,
                        selection='fixed_final_epoch'), checkpoint)
        actual_labels, values = scores(model, test_loader, device)
        if actual_labels != test_labels:
            raise RuntimeError('Test prediction order changed')
        fold_results.append((fold, test_pairs, test_labels, values))
        shared_predictions.extend(prediction_rows(
            fold, test_pairs, test_labels, values,
            float(settings['threshold']), epochs))
    write_tsv(shared / 'outer_oof_predictions.tsv', shared_predictions)
    return fold_results


def score_threshold(fold_results, threshold, epochs, root):
    directory = root / threshold_tag(threshold)
    directory.mkdir()
    summaries, predictions = [], []
    for fold, test_pairs, test_labels, values in fold_results:
        fold_dir = directory / f'fold_{fold}'
        fold_dir.mkdir()
        summaries.append(dict(fold=fold, epoch=epochs,
                              evaluation_threshold=threshold,
                              **metrics(test_labels, values, threshold)))
        current = prediction_rows(fold, test_pairs, test_labels, values,
                                  threshold, epochs)
        predictions.extend(current)
        write_tsv(fold_dir / f'fold_{fold}_predictions.tsv', current)
    write_tsv(directory / 'outer_per_fold_metrics.tsv', summaries)
    write_tsv(directory / 'outer_oof_predictions.tsv', predictions)
    pooled = metrics([row['True_Label'] for row in predictions],
                     [row['Predicted_Probability'] for row in predictions],
                     threshold)
    summary = [dict(threshold=threshold, metric=name,
                    mean=float(np.mean([row[name] for row in summaries])),
                    sd=float(np.std([row[name] for row in summaries], ddof=1)),
                    pooled=pooled[name])
               for name in METRIC_NAMES]
    write_tsv(directory / 'outer_summary.tsv', summary)
    return summary

