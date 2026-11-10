"""Independent architecture and checkpoint loader for V4."""
import torch
import segmentation_models_pytorch as smp


def build_v4(pretrained=False):
    return smp.DeepLabV3Plus(
        encoder_name='resnet18',
        encoder_weights='imagenet' if pretrained else None,
        encoder_output_stride=16,
        decoder_channels=256,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_v4(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('architecture') != 'DeepLabV3Plus-resnet18':
        raise ValueError('El checkpoint no corresponde a DeepLabV3+ V4.')
    model = build_v4(pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    return model, int(ckpt['size'])
