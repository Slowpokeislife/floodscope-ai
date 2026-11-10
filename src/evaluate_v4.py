"""One-time test of V4 using its threshold selected on validation."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import FloodZipDataset, region_of
from .metrics import confusion, summarize_counts
from .model_v4 import load_v4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/v4_deeplab_best.pt')
    parser.add_argument('--threshold-json', default='results/v4_deeplab_thresholds.json')
    parser.add_argument('--splits', default='results/splits.json')
    parser.add_argument('--data', default='data/flood_data.zip')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', default='results/v4_deeplab_test.json')
    args = parser.parse_args()
    tuned = json.loads(Path(args.threshold_json).read_text(encoding='utf-8'))
    if tuned.get('split') != 'val' or Path(tuned.get('checkpoint', '')).as_posix().lower() != Path(args.checkpoint).as_posix().lower():
        raise ValueError('Threshold JSON must come from validation of this exact checkpoint.')
    threshold = float(tuned['best_threshold'])
    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size = load_v4(args.checkpoint, device)
    loader = DataLoader(FloodZipDataset(args.data, splits['test'], size=size),
                        batch_size=args.batch_size, num_workers=0)
    pooled = [0, 0, 0, 0]
    per_region = {}
    with torch.inference_mode():
        for x, y, ids in loader:
            logits = model(x.to(device))
            y = y.to(device)
            for i, sample_id in enumerate(ids):
                c = confusion(logits[i:i+1], y[i:i+1], threshold)
                pooled = [a+b for a,b in zip(pooled, c)]
                region = region_of(sample_id)
                per_region.setdefault(region, [0, 0, 0, 0])
                per_region[region] = [a+b for a,b in zip(per_region[region], c)]
    output = {'checkpoint': args.checkpoint, 'threshold': threshold,
              'samples': len(loader.dataset), 'global': summarize_counts(*pooled),
              'by_region': {r: summarize_counts(*c) for r,c in sorted(per_region.items())},
              'notes': 'Pooled pixels; no-data not handled; DO NOT tune hyperparameters on test.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
