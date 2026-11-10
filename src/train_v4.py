"""V4: DeepLabV3+/ResNet18, same training protocol/split as V2."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodZipDataset, discover, region_of
from .metrics import bce_dice_loss, confusion, summarize_counts
from .model_v4 import build_v4


def epoch_pass(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    counts = [0, 0, 0, 0]
    total_loss = 0.0
    for x, y, _ in tqdm(loader, leave=False, desc='Train' if training else 'Val'):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = bce_dice_loss(logits.float(), y)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.detach().item() * len(x)
        result = confusion(logits.detach().float(), y, threshold=0.5)
        counts = [a + b for a, b in zip(counts, result)]
    result = summarize_counts(*counts)
    result['loss'] = total_loss / len(loader.dataset)
    return result


def main():
    parser = argparse.ArgumentParser(description='FloodScope V4: DeepLabV3+ 512x512')
    parser.add_argument('--data', default='data/flood_data.zip')
    parser.add_argument('--splits', default='results/splits.json')
    parser.add_argument('--size', type=int, default=512, choices=[256, 512])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--out', default='checkpoints/v4_deeplab_best.pt')
    parser.add_argument('--history', default='results/v4_deeplab_history.json')
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience) <= 0:
        parser.error('batch-size, epochs y patience deben ser positivos.')
    if Path(args.out).exists():
        parser.error(f'El checkpoint {args.out} ya existe. Especifica otro --out.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))
    known = set(discover(args.data))
    for part in ('train', 'val', 'test'):
        ids = splits.get(part, [])
        if not ids or not set(ids).issubset(known):
            raise ValueError(f'Split invalido: {part}')
        print(part, len(ids), sorted({region_of(i) for i in ids}))
    if any(set(splits[a]) & set(splits[b]) for a, b in
           [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Hay IDs compartidos entre splits.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = FloodZipDataset(args.data, splits['train'], size=args.size, augment=True)
    val_ds = FloodZipDataset(args.data, splits['val'], size=args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    model = build_v4(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best = -1.0
    stale = 0
    history = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    print('Architecture: DeepLabV3+ / ResNet18; device:', device, 'size:', args.size)
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_pass(model, train_loader, device, optimizer, scaler)
        val_metrics = epoch_pass(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics, 'lr': lr})
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | val Dice {val_metrics['dice']:.4f} | "
              f"val IoU {val_metrics['iou']:.4f} | lr {lr:.2e}", flush=True)
        if val_metrics['dice'] > best + 1e-4:
            best = val_metrics['dice']
            stale = 0
            torch.save({
                'state_dict': model.state_dict(), 'architecture': 'DeepLabV3Plus-resnet18',
                'size': args.size, 'epoch': epoch, 'validation_dice': best,
                'pretrained': args.pretrained, 'experiment': 'v4_deeplab_bce_dice',
                'seed': args.seed,
            }, args.out)
            print('Saved best:', args.out)
        else:
            stale += 1
        Path(args.history).write_text(
            json.dumps({'config': vars(args), 'history': history, 'best_validation_dice_050': best}, indent=2),
            encoding='utf-8')
        scheduler.step(val_metrics['dice'])
        if stale >= args.patience:
            print(f'Early stopping: {stale} epochs without improvement')
            break
    print(f'Best validation Dice at threshold 0.50: {best:.4f}; checkpoint: {args.out}')


if __name__ == '__main__':
    main()
