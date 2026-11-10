"""A segmentation U-Net with a ResNet-18 encoder."""
import segmentation_models_pytorch as smp


def build_model(pretrained=False):
    return smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )
