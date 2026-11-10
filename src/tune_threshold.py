"""Select a segmentation threshold on validation only, never on test.
Run: python -m src.tune_threshold --checkpoint checkpoints/best_model.pt
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodZipDataset, region_of
from .inference import load_checkpoint
from .metrics import summarize_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/flood_data.zip')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--splits', default='results/splits.json')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[round(i / 20, 2) for i in range(1, 20)])
    parser.add_argument('--output', default='results/threshold_search.json')
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        parser.error('Thresholds must be between 0 and 1, exclusive')

    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_checkpoint(args.checkpoint, device)
    ds = FloodZipDataset(args.data, splits['val'], size=size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # Counts are TP, FP, FN, TN, pooled across every pixel and every validation image.
    global_counts = {t: [0, 0, 0, 0] for t in thresholds}
    regional_counts = {t: {} for t in thresholds}

    print(f'Device: {device}; validation images: {len(ds)}; thresholds: {len(thresholds)}')
    with torch.inference_mode():
        for x, y, sample_ids in tqdm(loader, desc='Evaluating validation'):
            probs = torch.sigmoid(model(x.to(device))).cpu()
            target = y >= 0.5
            for t in thresholds:
                predicted = probs >= t
                for i, sample_id in enumerate(sample_ids):
                    p, truth = predicted[i], target[i]
                    c = [int((p & truth).sum()), int((p & ~truth).sum()),
                         int((~p & truth).sum()), int((~p & ~truth).sum())]
                    global_counts[t] = [a + b for a, b in zip(global_counts[t], c)]
                    region = region_of(sample_id)
                    rc = regional_counts[t].setdefault(region, [0, 0, 0, 0])
                    regional_counts[t][region] = [a + b for a, b in zip(rc, c)]

    rows = []
    for t in thresholds:
        row = {'threshold': t, **summarize_counts(*global_counts[t]),
               'by_region': {r: summarize_counts(*c)
                             for r, c in sorted(regional_counts[t].items())}}
        rows.append(row)
    # Optimize pooled validation Dice. Tie-breaking favors threshold closer to 0.5.
    best = max(rows, key=lambda r: (r['dice'], -abs(r['threshold'] - 0.5)))
    output = {'checkpoint': args.checkpoint, 'split': 'val', 'samples': len(ds),
              'selection_metric': 'pooled_validation_dice',
              'best_threshold': best['threshold'], 'best_validation_metrics':
              {k: v for k, v in best.items() if k != 'by_region'},
              'results': rows,
              'warning': 'Selected using validation data only. Do not tune on test.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('\nThreshold | Dice   | IoU    | Precision | Recall')
    for r in rows:
        print(f"{r['threshold']:8.2f} | {r['dice']:.4f} | {r['iou']:.4f} | "
              f"{r['precision']:.4f}   | {r['recall']:.4f}")
    print(f"\nBest threshold: {best['threshold']:.2f}; validation Dice: {best['dice']:.4f}")
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
