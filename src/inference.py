"""Shared inference and visualization helpers."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .model import build_model


def load_checkpoint(path='checkpoints/best_model.pt', device='cpu'):
    # Only load checkpoints that you created or trust.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['size']


def predict(model, pil_image, size, device='cpu', threshold=0.5):
    image = pil_image.convert('RGB')
    input_img = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(input_img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    # Resize continuous probabilities before thresholding.
    prob_img = Image.fromarray(probs.astype('float32'), mode='F')
    prob_img = prob_img.resize(image.size, Image.Resampling.BILINEAR)
    prob = np.asarray(prob_img)
    return (prob >= threshold).astype(np.uint8), prob


def overlay(image, mask, alpha=0.5):
    base = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    color = np.array([20, 160, 255], dtype=np.float32)
    base[mask == 1] = (1 - alpha) * base[mask == 1] + alpha * color
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
