"""FloodScope AI — Etapa 3A: reproducible validation diagnostics.

Run from project root:
 python -m src.analyze_errors --checkpoint checkpoints/v2_512_best.pt \
   --threshold-json results/v2_512_thresholds.json

Uses validation only for predictions; descriptive train/val/test mask prevalence is
computed only if --audit-all-splits (not needed for the normal workflow).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodZipDataset, discover, region_of
from .inference import load_checkpoint
from .metrics import summarize_counts


def counts(p, y):
    p, y = p.astype(bool), y.astype(bool)
    return (int(np.count_nonzero(p & y)), int(np.count_nonzero(p & ~y)),
            int(np.count_nonzero(~p & y)), int(np.count_nonzero(~p & ~y)))


def merge(a, b):
    return [int(x + z) for x, z in zip(a, b)]


def fmt(v):
    return f'{v:.3f}'


def make_gallery_item(img, target, prob, threshold, name):
    """All panels align in model-resolution coordinates; red=FP, yellow=FN, green=TP."""
    rgb = np.asarray(img, dtype=np.uint8)
    truth = target.astype(bool)
    predicted = prob >= threshold
    im = Image.fromarray(rgb).convert('RGB')
    gt = Image.fromarray((truth * 255).astype(np.uint8), 'L').convert('RGB')
    pred = Image.fromarray((predicted * 255).astype(np.uint8), 'L').convert('RGB')
    errors = np.zeros_like(rgb, dtype=np.uint8)
    errors[predicted & ~truth] = [245, 65, 70]     # FP
    errors[~predicted & truth] = [250, 195, 35]    # FN
    errors[predicted & truth] = [75, 190, 110]     # TP
    errors[~predicted & ~truth] = [30, 35, 45]    # TN
    panel_images = [im, gt, pred, Image.fromarray(errors)]
    width, height = im.size
    gap, head = 8, 52
    out = Image.new('RGB', (4 * width + 3 * gap, height + head), (245, 246, 248))
    d = ImageDraw.Draw(out)
    d.text((9, 6), f'{name}  |  threshold={threshold:.2f}', fill=(15, 20, 28))
    for i, (p, label) in enumerate(zip(panel_images, ['Input', 'Ground truth', 'Prediction', 'TP green / FP red / FN yellow'])):
        x = i * (width + gap)
        d.text((x + 6, 29), label, fill=(25, 30, 40))
        out.paste(p, (x, head))
    return out


def load_threshold(path, checkpoint):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = str(Path(checkpoint)).replace('\\', '/')
    actual = str(obj.get('checkpoint', '')).replace('\\', '/')
    if obj.get('split') != 'val' or actual != expected:
        raise ValueError('The threshold JSON must be selected on validation for THIS checkpoint.')
    return float(obj['best_threshold'])


def main():
    p = argparse.ArgumentParser(description='Error analysis on validation, not test')
    p.add_argument('--checkpoint', default='checkpoints/v2_512_best.pt')
    p.add_argument('--threshold-json', default='results/v2_512_thresholds.json')
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--splits', default='results/splits.json')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--gallery-count', type=int, default=4)
    p.add_argument('--outdir', default='results/error_analysis_v2')
    args = p.parse_args()
    if args.batch_size < 1 or args.gallery_count < 0:
        p.error('batch-size must be >= 1 and gallery-count >= 0')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    known = set(discover(args.data))
    for key in ('train', 'val', 'test'):
        if not split.get(key) or not set(split[key]) <= known:
            raise ValueError(f'Invalid/missing split: {key}')
    if any(set(split[a]) & set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Overlapping split IDs!')
    # Never run model inference on test here; use only validation.
    threshold = load_threshold(args.threshold_json, args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size = load_checkpoint(args.checkpoint, device)
    ds = FloodZipDataset(args.data, split['val'], size=size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Device={device}; split=validation; samples={len(ds)}; size={size}; threshold={threshold:.2f}')

    per_image = []
    region_counts = defaultdict(lambda: [0, 0, 0, 0])
    total_counts = [0, 0, 0, 0]
    save_candidates = []
    per_region = defaultdict(list)
    channel_sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    channel_n = defaultdict(int)

    with torch.inference_mode():
        for x, y, ids in tqdm(loader, desc='Analyzing validation'):
            prob_batch = torch.sigmoid(model(x.to(device))).float().cpu().numpy()[:, 0]
            x_np = x.numpy().transpose(0, 2, 3, 1)
            y_np = y.numpy()[:, 0] >= 0.5
            for i, sample_id in enumerate(ids):
                reg = region_of(sample_id)
                prob = prob_batch[i]
                truth = y_np[i]
                predicted = prob >= threshold
                c = counts(predicted, truth)
                total_counts = merge(total_counts, c)
                region_counts[reg] = merge(region_counts[reg], c)
                m = summarize_counts(*c)
                truth_frac = float(truth.mean())
                pred_frac = float(predicted.mean())
                row = dict(sample_id=sample_id, region=reg,
                           positive_fraction=truth_frac, predicted_fraction=pred_frac,
                           tp=c[0], fp=c[1], fn=c[2], tn=c[3], **m)
                per_image.append(row)
                per_region[reg].append(row)
                rgb = np.clip(x_np[i] * 255, 0, 255).astype(np.uint8)
                channel_sums[reg] += x_np[i].mean(axis=(0, 1))
                channel_n[reg] += 1
                # Keep only worst examples per region (bounded memory).
                item = (float(m['dice']), sample_id, rgb, truth.copy(), prob.copy())
                candidates = save_candidates
                candidates.append((reg, item))

    per_image.sort(key=lambda r: (r['region'], r['dice'], r['sample_id']))
    with (out / 'validation_per_image.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    regions = {}
    for r, vals in sorted(per_region.items()):
        pooled = summarize_counts(*region_counts[r])
        regions[r] = {'images': len(vals), 'pooled': pooled,
                      'mean_image_dice': float(np.mean([v['dice'] for v in vals])),
                      'mean_ground_truth_flood_fraction': float(np.mean([v['positive_fraction'] for v in vals])),
                      'mean_predicted_flood_fraction': float(np.mean([v['predicted_fraction'] for v in vals])),
                      'mean_rgb_channels_0to1': list((channel_sums[r] / channel_n[r]).round(5))}
    summary = {'checkpoint': args.checkpoint, 'threshold': threshold, 'split': 'val',
               'samples': len(ds), 'image_size': size,
               'global_pooled': summarize_counts(*total_counts),
               'by_region': regions,
               'warnings': [
                   'These are validation diagnostics; do not use test samples for model selection.',
                   'RGB values are stored PNG channels; their physical SAR meaning is not established.',
                   'Dataset maps mask 255->flood and every other value->nonflood; original no-data semantics require verification.',
                   'Per-image Dice of empty truth and empty prediction is reported as zero by confusion-based formula; inspect prevalence alongside it.',
               ]}
    (out / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nREGION | samples | pooled Dice | pooled Precision | pooled Recall | GT flooded % | Pred flooded %')
    for reg, stats in regions.items():
        m = stats['pooled']
        print(f"{reg:12} {stats['images']:>5}   {m['dice']:.4f}       {m['precision']:.4f}           {m['recall']:.4f}      "
              f"{100*stats['mean_ground_truth_flood_fraction']:.2f}           {100*stats['mean_predicted_flood_fraction']:.2f}")

    gallery = out / 'gallery'
    gallery.mkdir(exist_ok=True)
    for reg in sorted(regions):
        candidates = sorted([item for r,item in save_candidates if r == reg], key=lambda item: (item[0], item[1]))
        for ix, (score, sid, rgb, truth, prob) in enumerate(candidates[:args.gallery_count], 1):
            make_gallery_item(rgb, truth, prob, threshold, sid).save(gallery / f'{reg}_{ix:02d}_{sid}.png')
    print(f'\nSaved: {out / "validation_per_image.csv"}')
    print(f'Saved: {out / "validation_summary.json"}')
    print(f'Gallery: {gallery}')
    print('Note: gallery shows worst Dice examples from validation, not test.')


if __name__ == '__main__':
    main()
