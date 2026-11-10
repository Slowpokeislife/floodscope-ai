"""One-time final test with a threshold chosen ONLY on validation."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from .dataset import FloodZipDataset, region_of
from .inference import load_checkpoint
from .metrics import confusion, summarize_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--threshold-json', required=True, help='Output from tune_threshold on validation')
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--splits', default='results/splits.json')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    tune = json.loads(Path(args.threshold_json).read_text(encoding='utf8'))
    if tune.get('split') != 'val' or tune.get('checkpoint', '').replace('\\', '/') != args.checkpoint.replace('\\', '/'):
        raise ValueError('Threshold must be selected on validation from exactly this checkpoint')
    threshold = float(tune['best_threshold'])
    splits = json.loads(Path(args.splits).read_text(encoding='utf8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(FloodZipDataset(args.data, splits['test'], size=size),
                        batch_size=args.batch_size, num_workers=0)
    pooled, regions = [0, 0, 0, 0], {}
    with torch.inference_mode():
        for x, y, names in loader:
            logits = model(x.to(device))
            y = y.to(device)
            for i, name in enumerate(names):
                c = confusion(logits[i:i+1], y[i:i+1], threshold=threshold)
                pooled = [a+b for a,b in zip(pooled,c)]
                region = region_of(name)
                regions.setdefault(region, [0,0,0,0])
                regions[region] = [a+b for a,b in zip(regions[region],c)]
    results = {'checkpoint': args.checkpoint, 'threshold': threshold, 'samples': len(splits['test']),
               'global': summarize_counts(*pooled),
               'by_region': {k: summarize_counts(*v) for k,v in sorted(regions.items())},
               'notes': 'Pooled pixels; ignores no-data semantics; test must not be used for tuning.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
