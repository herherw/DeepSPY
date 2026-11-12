"""Fixed-final-epoch training on saved, ordered cross-validation splits."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
from pathlib import Path


def load_splits(pair_path):
    with pair_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    pairs = [r['pair_id'] for r in rows]
    labels = [int(r['label']) for r in rows]
    canonical = dict(zip(pairs, labels))
    index = {p: i for i, p in enumerate(pairs)}
    if len(canonical) != len(pairs) or set(labels) != {0, 1}:
        raise ValueError('Expected unique pairs and both classes')
    splits, tested = [], set()
    for fold in range(1, 6):
        groups = []
        for split in ('train', 'test'):
            file = pair_path.parent / 'splits' / f'fold_{fold}' / f'{split}_pairs.txt'
            entries = [line.split() for line in file.read_text(encoding='utf-8').splitlines()]
            if len({p for p, _ in entries}) != len(entries) or any(canonical.get(p) != int(y) for p, y in entries):
                raise ValueError(f'Invalid pairs or labels in {file}')
            groups.append([index[p] for p, _ in entries])
        train, test = map(set, groups)
        if train & test or train | test != set(index.values()) or test & tested:
            raise ValueError(f'Inconsistent fold {fold}')
        tested.update(test)
        splits.append(tuple(groups))
    if tested != set(index.values()):
        raise ValueError('Every pair must be tested once')
    return pairs, labels, splits


def main(config_path, validate_only=False, device_name=None):
    from .utils import get_config, save_source_code
    config_path = Path(config_path).resolve()
    config = get_config(str(config_path))
    pairs, labels, splits = load_splits(Path(config['data']['pair_path']))
    print('Test fold sizes:', [len(test) for _, test in splits], flush=True)
    if validate_only:
        return
    import torch
    from .fixed_training import score_threshold, train_once, validate_thresholds
    settings = config['train']
    thresholds = validate_thresholds(settings['threshold_candidates'])
    epochs = config.get('sensitivity_epochs', [settings['epoch']])
    if not epochs or len(set(epochs)) != len(epochs) or any(type(e) is not int or e < 1 for e in epochs):
        raise ValueError('Epoch counts must be unique positive integers')
    device = torch.device(device_name or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    root = Path(config['output']['model_save_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    for epoch in epochs:
        current = deepcopy(config)
        current['train']['epoch'] = epoch
        folder = root if len(epochs) == 1 else root / f'epoch_{epoch}'
        folder.mkdir(exist_ok=True)
        save_source_code(str(folder), str(config_path), current, str(Path(__file__).parent))
        values = train_once(current, pairs, labels, splits, folder, device)
        for threshold in thresholds:
            score_threshold(values, threshold, epoch, folder)
    print(f'Completed: {root}')
    return root


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--device', help='Override the automatically selected torch device')
    args = parser.parse_args()
    main(args.config, args.validate_only, args.device)


if __name__ == '__main__':
    cli()
