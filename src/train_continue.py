"""Fine-tune from existing best model without changing original checkpoint or split.
Run: python -m src.train_continue --epochs 30 --lr 3e-5
This is weight-only continuation; previous Adam state was not saved.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FloodZipDataset
from .inference import load_checkpoint
from .train import run_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--splits', default='results/splits.json')
    p.add_argument('--epochs', type=int, default=30, help='Additional epochs, not total')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--output', default='checkpoints/extended_best_model.pt')
    p.add_argument('--history', default='results/extended_history.json')
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = load_checkpoint(args.checkpoint, device)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    starting_epoch = int(source.get('epoch', 0))
    print(f'Loaded {args.checkpoint}; model size={size}, source epoch={starting_epoch}')
    print(f'Device: {device}; weight-only continuation (Adam optimizer state reset)')
    if len(splits['train']) == 0 or len(splits['val']) == 0:
        raise ValueError('Empty train or val split')
    train_ds = FloodZipDataset(args.data, splits['train'], size=size, augment=True)
    val_ds = FloodZipDataset(args.data, splits['val'], size=size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Recompute baseline on validation so best checkpoint is always an actual validation improvement.
    baseline = run_epoch(model, val_loader, device)
    best_dice = baseline['dice']
    history = [{'additional_epoch': 0, 'epoch': starting_epoch,
                'val': baseline, 'checkpoint': args.checkpoint}]
    bad_epochs = 0
    print(f'Baseline validation Dice at 0.50 threshold: {best_dice:.4f}')
    out = Path(args.output)
    if out.resolve() == Path(args.checkpoint).resolve():
        raise ValueError('Output must differ from original checkpoint path')
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)

    for extra_epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        print(f"Additional epoch {extra_epoch}/{args.epochs}: "
              f"train loss={train['loss']:.4f}, val loss={val['loss']:.4f}, "
              f"val Dice={val['dice']:.4f}, val IoU={val['iou']:.4f}")
        history.append({'additional_epoch': extra_epoch,
                        'epoch': starting_epoch + extra_epoch,
                        'train': train, 'val': val})
        if val['dice'] > best_dice + 1e-5:
            best_dice = val['dice']
            bad_epochs = 0
            torch.save({'state_dict': model.state_dict(), 'size': size,
                        'epoch': starting_epoch + extra_epoch,
                        'validation_dice': best_dice,
                        'split_method': source.get('split_method', 'unknown'),
                        'continued_from': args.checkpoint,
                        'additional_epoch': extra_epoch}, out)
            print(f'  Improved! Saved {out}')
        else:
            bad_epochs += 1
        Path(args.history).write_text(json.dumps(history, indent=2), encoding='utf-8')
        if bad_epochs >= args.patience:
            print(f'Early stopping after {bad_epochs} epochs without improvement')
            break
    print(f'Best additional-training val Dice: {best_dice:.4f}')
    if not out.exists():
        print('No improvement. Keep using original checkpoint:', args.checkpoint)
    else:
        print('Extended checkpoint:', out)


if __name__ == '__main__':
    main()
