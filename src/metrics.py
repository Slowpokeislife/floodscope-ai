import torch
import torch.nn.functional as F


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    dice = (2 * (probs * target).sum(dim=dims) + 1) / (
        probs.sum(dim=dims) + target.sum(dim=dims) + 1
    )
    return bce + (1 - dice).mean()


@torch.no_grad()
def confusion(logits, target, threshold=0.5):
    p = torch.sigmoid(logits) >= threshold
    t = target >= 0.5
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    return tp, fp, fn, tn


def summarize_counts(tp, fp, fn, tn):
    eps = 1e-8
    return {
        'iou': tp / (tp + fp + fn + eps),
        'dice': 2 * tp / (2 * tp + fp + fn + eps),
        'precision': tp / (tp + fp + eps),
        'recall': tp / (tp + fn + eps),
        'pixel_accuracy': (tp + tn) / (tp + fp + fn + tn + eps),
    }
