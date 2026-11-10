"""Optimize DeepLabV3+ threshold on validation ONLY."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import FloodZipDataset, region_of
from .metrics import summarize_counts
from .model_v4 import load_v4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/v4_deeplab_best.pt')
    parser.add_argument('--data', default='data/flood_data.zip')
    parser.add_argument('--splits', default='results/splits.json')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', default='results/v4_deeplab_thresholds.json')
    args = parser.parse_args()
    thresholds = [round(i / 20, 2) for i in range(1, 20)]
    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size = load_v4(args.checkpoint, device)
    loader = DataLoader(FloodZipDataset(args.data, splits['val'], size=size),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    counts = {t: [0, 0, 0, 0] for t in thresholds}
    print('Device:', device, 'validation images:', len(loader.dataset))
    with torch.inference_mode():
        for x, y, _ in tqdm(loader, desc='Validation thresholds'):
            prob = torch.sigmoid(model(x.to(device))).cpu()
            truth = y >= 0.5
            for threshold in thresholds:
                pred = prob >= threshold
                c = [int((pred & truth).sum()), int((pred & ~truth).sum()),
                     int((~pred & truth).sum()), int((~pred & ~truth).sum())]
                counts[threshold] = [a+b for a,b in zip(counts[threshold], c)]
    rows = [{'threshold': t, **summarize_counts(*counts[t])} for t in thresholds]
    best = max(rows, key=lambda r: (r['dice'], -abs(r['threshold']-0.5)))
    payload = {'checkpoint': args.checkpoint, 'split': 'val', 'samples': len(loader.dataset),
               'best_threshold': best['threshold'], 'best_validation_metrics': best,
               'results': rows, 'architecture': 'DeepLabV3Plus-resnet18'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('\nThreshold | Dice   | IoU    | Precision | Recall')
    for r in rows:
        print(f"{r['threshold']:8.2f} | {r['dice']:.4f} | {r['iou']:.4f} | "
              f"{r['precision']:.4f}    | {r['recall']:.4f}")
    print(f"Best threshold: {best['threshold']:.2f}; validation Dice: {best['dice']:.4f}")
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
