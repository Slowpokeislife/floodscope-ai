"""Exploratory Nepal inference. NOT validated flood mapping.

Run from project root: python -m src.nepal_exploratory ...
Depends on existing src/model_v4.py, torch, pillow, numpy, matplotlib.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from src.model_v4 import load_v4

DEFAULT_TILES = ["tile_12_12.png", "tile_26_22.png", "tile_14_28.png", "tile_34_46.png"]


def read_gray(z, name):
    with z.open(name) as f:
        im = Image.open(BytesIO(f.read()))
        if im.mode != 'L':
            raise ValueError(f"Expected grayscale PNG, got {im.mode}: {name}")
        return np.asarray(im, dtype=np.uint8)


def compose(vv_u8, vh_u8, filtered=True):
    # Reproduces the *documented Nepal case study* composite, NOT a
    # confirmed mapping to Sen1Floods11 training PNGs.
    vv = vv_u8.astype(np.float32) / 255.0
    vh = vh_u8.astype(np.float32) / 255.0
    r = vv.copy()
    if filtered:
        r[r > 0.7] = 0.0
    g = vh
    b = np.clip(np.divide(vv, vh, out=np.zeros_like(vv), where=vh > 0), 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


@torch.inference_mode()
def predict(model, rgb, device, model_size):
    if rgb.shape[:2] != (model_size, model_size):
        # Model trained at 512, so resize like dataset.py if necessary.
        im = Image.fromarray((rgb * 255).round().astype(np.uint8), mode='RGB')
        im = im.resize((model_size, model_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.sigmoid(logits[0, 0]).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description='Exploratory Nepal SAR inference, NOT validated flood detection.')
    ap.add_argument('--nepal', default='data/UNOSAT_case_study.zip')
    ap.add_argument('--checkpoint', default='checkpoints/v4_deeplab_best.pt')
    ap.add_argument('--outdir', default='results/nepal_exploratory')
    ap.add_argument('--tiles', nargs='+', default=DEFAULT_TILES)
    ap.add_argument('--threshold', type=float, default=0.45, help='Diagnostic display only; do not tune on Nepal')
    args = ap.parse_args()
    if not 0 < args.threshold < 1:
        ap.error('--threshold must be between 0 and 1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size = load_v4(args.checkpoint, device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    with ZipFile(args.nepal) as z:
        names = set(z.namelist())
        for tile in args.tiles:
            tile = Path(tile).name
            vvname, vhname = f'tiles_VV/{tile}', f'tiles_VH/{tile}'
            if vvname not in names or vhname not in names:
                print(f'SKIP missing VV/VH pair: {tile}')
                entries.append({'tile': tile, 'status':'missing_pair'})
                continue
            vv, vh = read_gray(z, vvname), read_gray(z, vhname)
            if vv.shape != vh.shape:
                print(f'SKIP shape mismatch: {tile}')
                entries.append({'tile': tile, 'status':'shape_mismatch'})
                continue
            # Exploratory warning only, NOT a verified no-data mask.
            suspect = (vv == 255) | (vh == 0)
            suspect_fraction = float(suspect.mean())
            if suspect_fraction >= 0.98:
                print(f'SKIP suspected degenerate tile: {tile}, suspect={suspect_fraction:.1%}')
                entries.append({'tile':tile, 'status':'suspect_degenerate', 'suspect_fraction':suspect_fraction})
                continue
            filtered = compose(vv, vh, filtered=True)
            plain = compose(vv, vh, filtered=False)
            pf = predict(model, filtered, device, size)
            pp = predict(model, plain, device, size)
            np.savez_compressed(out / f'{Path(tile).stem}_probabilities.npz',
                                filtered=pf, unfiltered=pp)
            fig, ax = plt.subplots(1, 5, figsize=(17, 3.7))
            ax[0].imshow(filtered); ax[0].set_title('RGB, filtro VV')
            ax[1].imshow(plain); ax[1].set_title('RGB, sin filtro VV')
            ax[2].imshow(pf, vmin=0, vmax=1, cmap='viridis'); ax[2].set_title('Probabilidad, filtro VV')
            ax[3].imshow(pp, vmin=0, vmax=1, cmap='viridis'); ax[3].set_title('Probabilidad, sin filtro')
            ax[4].imshow(pf >= args.threshold, cmap='gray', vmin=0, vmax=1)
            ax[4].set_title(f'Máscara exploratoria, t={args.threshold:g}')
            for a in ax:
                a.axis('off')
            fig.suptitle(f'{tile} | regiones sospechosas (regla no validada): {suspect_fraction:.1%}', fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f'{Path(tile).stem}_comparison.png', dpi=140, bbox_inches='tight')
            plt.close(fig)
            entries.append({
                'tile': tile, 'status':'ok', 'suspect_fraction':suspect_fraction,
                'mean_probability_filtered':float(pf.mean()),
                'mean_probability_unfiltered':float(pp.mean()),
                'positive_fraction_filtered':float((pf >= args.threshold).mean()),
                'positive_fraction_unfiltered':float((pp >= args.threshold).mean()),
                'mean_abs_probability_change':float(np.abs(pf-pp).mean()),
            })
            print(f'OK {tile}: suspect={suspect_fraction:.1%}, mean abs prob delta={np.abs(pf-pp).mean():.3f}')
    report = {
      'status':'EXPLORATORY_ONLY_NOT_VALIDATED',
      'warning':'No ground truth for Nepal; no reliable flood area or population estimates.',
      'nepal_input':'VV/VH PNG/255, composite from UNOSAT notebook; compatibility with training RGB NOT established',
      'input_pipeline':'RGB [0,1] CHW (matches src/dataset.py format, not necessarily SAR semantics)',
      'checkpoint':args.checkpoint, 'threshold_visualization_only':args.threshold,
      'checkpoint_model_size':size, 'tiles':entries
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Finished. See {out / "report.json"}')


if __name__ == '__main__':
    main()
