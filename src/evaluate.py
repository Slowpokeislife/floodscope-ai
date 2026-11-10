"""Evaluate a saved model on the untouched test set."""
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
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()
    splits = json.loads(Path('results/splits.json').read_text(encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_checkpoint(args.checkpoint, device)
    ds = FloodZipDataset(args.data, splits['test'], size=size)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    total = [0, 0, 0, 0]
    regional = {}
    with torch.no_grad():
        for x, y, sample_ids in loader:
            out = model(x.to(device))
            for i, name in enumerate(sample_ids):
                counts = confusion(out[i:i+1], y[i:i+1].to(device))
                total = [a + b for a, b in zip(total, counts)]
                r = region_of(name)
                regional.setdefault(r, [0, 0, 0, 0])
                regional[r] = [a + b for a, b in zip(regional[r], counts)]
    metrics = {'global': summarize_counts(*total),
               'by_region': {r: summarize_counts(*c) for r, c in regional.items()},
               'samples': len(ds), 'notes': 'Pixel-level pooled metrics; excludes no-data special handling.'}
    Path('results/test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
