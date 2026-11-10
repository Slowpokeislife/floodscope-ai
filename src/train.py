"""Train and save a U-Net. Usage: python -m src.train --epochs 15"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import FloodZipDataset, discover, create_splits, region_of
from .model import build_model
from .metrics import bce_dice_loss, confusion, summarize_counts


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sum_loss, sum_counts = 0., [0, 0, 0, 0]
    for x, y, _ in tqdm(loader, leave=False, disable=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        sum_loss += float(loss.item()) * x.size(0)
        counts = confusion(logits.detach(), y)
        sum_counts = [a + b for a, b in zip(sum_counts, counts)]
    stats = summarize_counts(*sum_counts)
    stats['loss'] = sum_loss / len(loader.dataset)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--size', type=int, default=256, choices=[128, 256, 512])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['region', 'random'], default='region')
    p.add_argument('--pretrained', action='store_true', help='Download ImageNet ResNet-18 weights')
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    ids = discover(args.data)
    splits = create_splits(ids, seed=args.seed, method=args.split)
    for part, names in splits.items():
        print(f'{part}: {len(names)} images; regions={sorted({region_of(n) for n in names})}')
        if not names:
            raise ValueError(f'Empty split: {part}')
    Path('results/splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    loaders = {}
    for part in ('train', 'val'):
        ds = FloodZipDataset(args.data, splits[part], size=args.size, augment=part == 'train')
        loaders[part] = DataLoader(ds, batch_size=args.batch_size, shuffle=part == 'train', num_workers=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders['train'], device, optimizer)
        val = run_epoch(model, loaders['val'], device)
        print(f"Epoch {epoch}/{args.epochs} - train loss {train['loss']:.4f}, "
              f"val loss {val['loss']:.4f}, val Dice {val['dice']:.4f}, val IoU {val['iou']:.4f}")
        history.append({'epoch': epoch, 'train': train, 'val': val})
        if val['dice'] > best_dice:
            best_dice = val['dice']
            torch.save({'state_dict': model.state_dict(), 'size': args.size,
                        'epoch': epoch, 'validation_dice': best_dice,
                        'split_method': args.split}, 'checkpoints/best_model.pt')
        Path('results/history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('Best validation Dice:', round(best_dice, 4))
    print('Model saved to checkpoints/best_model.pt')


if __name__ == '__main__':
    main()
