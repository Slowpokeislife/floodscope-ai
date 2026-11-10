"""V3: train U-Net/ResNet18 with BCE+Tversky (or Focal Tversky)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodZipDataset, discover, region_of
from .model import build_model
from .metrics import confusion, summarize_counts
import torch.nn.functional as F


def segmentation_loss(logits, target, alpha=0.7, beta=0.3, loss_type="bce_tversky", gamma=1.33):
    """alpha weights false positives; beta weights false negatives.
    Tversky is computed per image to avoid domination by large masks.
    """
    logits = logits.float()
    probs = torch.sigmoid(logits)
    axes = (1, 2, 3)
    tp = (probs * target).sum(dim=axes)
    fp = (probs * (1 - target)).sum(dim=axes)
    fn = ((1 - probs) * target).sum(dim=axes)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    tversky_loss = (1.0 - score).clamp(0, 1)
    if loss_type == "focal_tversky":
        tversky_loss = tversky_loss.pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + tversky_loss.mean()



def epoch_pass(model, loader, device, loss_options, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    counts = [0, 0, 0, 0]
    total_loss = 0.0
    for x, y, _ in tqdm(loader, leave=False, desc='Train' if train else 'Val'):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                logits = model(x)
                loss = segmentation_loss(logits.float(), y, **loss_options)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.item()) * x.size(0)
        c = confusion(logits.detach().float(), y, threshold=0.5)
        counts = [a + b for a, b in zip(counts, c)]
    result = summarize_counts(*counts)
    result['loss'] = total_loss / len(loader.dataset)
    return result


def main():
    p = argparse.ArgumentParser(description='V3: 512px BCE + Tversky experiment, isolated from V1/V2')
    p.add_argument('--data', default='data/flood_data.zip')
    p.add_argument('--splits', default='results/splits.json')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--size', type=int, default=512, choices=[256, 512])
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--alpha', type=float, default=0.7, help='Penalty for false positives')
    p.add_argument('--beta', type=float, default=0.3, help='Penalty for false negatives')
    p.add_argument('--loss-type', choices=['bce_tversky', 'focal_tversky'], default='bce_tversky')
    p.add_argument('--gamma', type=float, default=1.33, help='Exponent used for focal_tversky only')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--out', default='checkpoints/v3_tversky_best.pt')
    p.add_argument('--history', default='results/v3_tversky_history.json')
    args = p.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        p.error('batch-size, epochs and patience must be positive')
    if not (0 < args.alpha < 1 and 0 < args.beta < 1) or abs(args.alpha + args.beta - 1) > 1e-6:
        p.error('alpha and beta must be positive and sum to 1')
    if args.gamma <= 0:
        p.error('gamma must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    splits = json.loads(Path(args.splits).read_text(encoding='utf8'))
    all_ids = set(discover(args.data))
    for part in ('train', 'val', 'test'):
        if not splits.get(part) or not set(splits[part]).issubset(all_ids):
            raise ValueError(f'Missing or invalid {part} IDs in {args.splits}')
        print(part, len(splits[part]), sorted({region_of(i) for i in splits[part]}))
    if set(splits['train']) & set(splits['val']) or set(splits['train']) & set(splits['test']) or set(splits['val']) & set(splits['test']):
        raise ValueError('Splits overlap')

    train_ds = FloodZipDataset(args.data, splits['train'], size=args.size, augment=True)
    val_ds = FloodZipDataset(args.data, splits['val'], size=args.size, augment=False)
    loaders = {
        'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                          pin_memory=torch.cuda.is_available()),
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, 'resolution:', args.size)
    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loss_options = {'alpha': args.alpha, 'beta': args.beta, 'loss_type': args.loss_type, 'gamma': args.gamma}
    print('Loss:', loss_options)
    best = -1.0
    no_improve = 0
    history = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        raise FileExistsError(f'{args.out} already exists. Choose another --out to avoid overwriting.')
    for epoch in range(1, args.epochs + 1):
        tr = epoch_pass(model, loaders['train'], device, loss_options, optimizer, scaler)
        va = epoch_pass(model, loaders['val'], device, loss_options)
        lr_used = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train': tr, 'val': va, 'lr': lr_used})
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} | "
              f"val Dice {va['dice']:.4f} | val IoU {va['iou']:.4f} | lr {lr_used:.2e}", flush=True)
        if va['dice'] > best + 1e-4:
            best = va['dice']
            no_improve = 0
            torch.save({'state_dict': model.state_dict(), 'size': args.size,
                        'epoch': epoch, 'validation_dice': best,
                        'split_method': 'reused_v1', 'experiment': 'v3_512_adamw_tversky',
                        'loss_type': args.loss_type, 'alpha': args.alpha, 'beta': args.beta,
                        'gamma': args.gamma, 'pretrained': args.pretrained}, args.out)
            print('Saved new best:', args.out)
        else:
            no_improve += 1
        Path(args.history).write_text(json.dumps({'config': vars(args), 'history': history}, indent=2), encoding='utf8')
        scheduler.step(va['dice'])
        if no_improve >= args.patience:
            print(f'Early stopping: {args.patience} consecutive epochs without improvement')
            break
    print(f'Best validation Dice at threshold 0.50: {best:.4f}; checkpoint: {args.out}')


if __name__ == '__main__':
    main()
