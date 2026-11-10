"""Prepare a conservative, separate GitHub staging copy of FloodScope AI.
Run from the ORIGINAL project root: python prepare_github.py
Never overwrites the original files. Python stdlib only.
"""
from pathlib import Path
import argparse
import shutil
import json
import sys

TOP_LEVEL_FILES = ['INSTRUCCIONES.md']
RESULT_FILES = [
    'splits.json','test_metrics.json','threshold_search.json',
    'v1_test_threshold060.json','v2_512_test.json',
    'v2_512_thresholds.json','v3_tversky_thresholds.json',
    'v4_deeplab_test.json','v4_deeplab_thresholds.json',
    'history.json','v2_512_history.json','v3_tversky_history.json','v4_deeplab_history.json',
]
RESULT_SUBDIRS = {
    'error_analysis_v1':['validation_summary.json'],
    'error_analysis_v2':['validation_summary.json'],
    'error_analysis_v3':['validation_summary.json'],
    'data_audit':['audit_report.json','RESUMEN.md'],
    'nepal_diagnostics':['report.json','RESUMEN.md'],
    'nepal_exploratory':['report.json'],
}
IGNORE = '''# Private/runtime artifacts
.venv/
venv/
.env
.env.*
!.env.example
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.streamlit/secrets.toml
.ipynb_checkpoints/
# Training data, proprietary course files, and model weights
data/*
!data/README.md
!data/.gitkeep
checkpoints/*
!checkpoints/README.md
!checkpoints/.gitkeep
*.pt
*.pth
*.ckpt
*.engine
*.onnx
*.tif
*.tiff
# Local artifacts
results/**/gallery/
results/**/*.npz
results/**/*.npy
results/**/validation_per_image.csv
results/**/sample_channel_stats.csv
results/**/nepal_tile_stats.csv
estructura.txt
estructura*.txt
.DS_Store
Thumbs.db
# Project-specific temporary scratch
*.log
'''
REQUIREMENTS = '''# PyTorch and torchvision must be installed separately for your CPU/CUDA platform.
# See README: PyTorch installation instructions.
numpy>=1.24,<3
Pillow>=10,<13
segmentation-models-pytorch>=0.4,<0.6
streamlit>=1.37,<2
pandas>=2.1
tqdm>=4.66
matplotlib>=3.8
'''
DATA_README = '''# Dataset acquisition

`flood_data.zip` and `UNOSAT_case_study.zip` are **not included** in this repository.

This research originated from an NVIDIA Deep Learning Institute disaster-risk exercise involving Sentinel-1 imagery and the Sen1Floods11 dataset. Obtain data from an authorized source and review applicable licenses/terms before reuse. For the existing scripts, place the authorized training archive at `data/flood_data.zip`. The data interface expects `images/all_images/*.png` and `masks/all_masks/*.png` paired by stem within that ZIP.

The UNOSAT/Nepal ZIP is an *exploratory domain-shift investigation only*: it is not necessary to reproduce labeled segmentation metrics. There is no validated Nepal ground truth in this project.

Do not interpret model probability maps as operational flood determinations.
'''
CKPT_README = '''# Checkpoints

Model checkpoints are intentionally excluded from source control.

Expected filenames for the interactive model explorer:

- `v2_512_best.pt` — U-Net/ResNet-18, 512×512.
- `v4_deeplab_best.pt` — DeepLabV3+/ResNet-18, 512×512.

The training code is included. Subject to rights review, trained weights may be provided as separate GitHub Release assets. Until then the dashboard can display documented metrics without running weight-dependent inference.

Note that weights derived from third-party data and training materials may have redistribution conditions. Verify them before releasing.
'''
DOC_METHOD = '''# Methodology and evaluation protocol

## Data

446 paired 512×512 RGB PNG images and binary masks were identified in the local research archive. Region-wise splits were used:

| Split | Count | Regions |
| --- | ---: | --- |
| Train | 292 | Bolivia, Ghana, Nigeria, Pakistan, Paraguay, Sri-Lanka, USA |
| Validation | 98 | India, Spain |
| Test | 56 | Mekong, Somalia |

Masks were audited and contained values 0 and 255. For model training, input RGB PNG values were normalized to [0,1] without ImageNet channel normalization. This **does not prove** that the physical meaning of RGB channels matches VV/VH composites used elsewhere.

## Model selection

Validation Dice determined checkpoint selection; candidate thresholds were selected **on validation**. The performance summaries below are pixel-pooled segmentation metrics, not mean image Dice.

**Important:** the test regions were inspected repeatedly throughout development. Comparisons on this historical test set are therefore *exploratory rather than a pristine final benchmark*. Further selection should use fresh held-out geographic regions/events.

## Failure analysis

The model underperforms on sparse positives, narrow structures, and geographic shifts; negative tiles can exhibit false-positive regions. The Nepal VV/VH-to-RGB composite yielded near-universal positive predictions (~98–100% of pixels in four example tiles), demonstrating a severe input-domain mismatch. It is **not** a valid flood extent or impact estimate.

## Limitations

- Special no-data semantics in original upstream imagery are not fully verified.
- Region counts are modest; results may not generalize to other dates, sensors, processing pipelines, or flood events.
- RGB training imagery and Nepal VV/VH composites are not proven semantically compatible.
- No verified Nepal ground truth or geospatial area/population-impact calculations.
- Small-object detection and false alarms require further systematic measurement.
'''
README = '''# FloodScope AI 🌊

**Satellite Flood Segmentation, Geographic Generalization & Domain Shift**

A computer-vision research project built around Sentinel-1-derived imagery. FloodScope AI compares U-Net and DeepLabV3+ segmentation pipelines, evaluates models across geographically separated regions, investigates difficult examples, and provides a Streamlit research dashboard.

> **Research demonstration, not an emergency response or operational flood-warning system.** Outputs are model predictions, not verified flood maps. The Nepal study illustrates a *failed cross-domain transfer*, not a validated estimate of inundation.

## What the project does

- Trains and evaluates four segmentation experiments with PyTorch.
- Uses region-disjoint training, validation, and test partitions.
- Compares Dice, IoU, precision, recall and pixel accuracy at region level.
- Tunes probability thresholds on validation only.
- Explores RGB image/mask examples and visualizes false positives/negatives.
- Documents the mismatch between RGB training data and Nepal VV/VH-composite imagery.
- Provides a Streamlit dashboard with model comparison, geographic metrics and error analysis.

## Results

| Version | Architecture / loss | Best validation Dice | Observed test Dice | Validation threshold |
| --- | --- | ---: | ---: | ---: |
| V1 | U-Net / BCE+Dice, 256² | 0.7198 | 0.6528 | 0.60 |
| V2 | U-Net / BCE+Dice, 512² | **0.7301** | 0.6799 | 0.40 |
| V3 | U-Net / BCE+Tversky, 512² | 0.7270 | Not evaluated | 0.60 |
| V4 | DeepLabV3+ / BCE+Dice, 512² | 0.7189 | **0.7185** | 0.45 |

*The test set was examined repeatedly during development. Treat test-set model ranking as exploratory, not definitive evidence of architectural superiority. Figures are pixel-pooled.*

### Geographical transfer

Held-out test Dice for U-Net V2: Mekong **0.8751**, Somalia **0.2711**. For DeepLabV3+ V4: Mekong **0.8648**, Somalia **0.2934**. Regional performance is substantially heterogeneous.

## Dashboard

The UI uses a warm-cream and electric-blue visual system with four research views: **Overview**, **Model Explorer**, **Geographic Evaluation**, and **Nepal — Domain Shift**.

![Dashboard screenshot](docs/screenshots/dashboard.png)

*Add your own dashboard screenshot at `docs/screenshots/dashboard.png` before publishing; alternatively remove the image link.*

## Quickstart (Windows / PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

Install a version of **PyTorch + torchvision** compatible with your device (CUDA or CPU) using the **official** selector: https://pytorch.org/get-started/locally/ . Then:

```powershell
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard's summary views can use saved `results/*.json` without weights. **Model Explorer** requires authorized training data and the matching model checkpoints; see `data/README.md` and `checkpoints/README.md`. Do not assume remote hosting with free CPU/RAM supports model inference or heavy datasets.

## Reproducing experiments

The following commands assume you obtained the dataset legally at `data/flood_data.zip` and run them from the project root:

```powershell
python -m src.train_v2 --epochs 50 --size 512 --batch-size 2 --lr 0.0001 --patience 10 --pretrained
python -m src.tune_threshold --checkpoint checkpoints/v2_512_best.pt --batch-size 2 --output results/v2_512_thresholds.json
python -m src.train_v4 --epochs 50 --size 512 --batch-size 2 --lr 0.0001 --patience 10 --pretrained
python -m src.tune_threshold_v4 --checkpoint checkpoints/v4_deeplab_best.pt --batch-size 2 --output results/v4_deeplab_thresholds.json
```

Metrics and inference are dependent on weights, exact environment, downloaded upstream encoder weights, and the chosen geographic split; randomness and GPU kernels may vary. See [methodology and limitations](docs/methodology.md).

## Repository map

```text
app/              Streamlit dashboard and styling
src/              datasets, models, training, evaluation, analysis
results/          selected JSON metrics and summaries
checkpoints/      instructions for obtaining model weights (not committed)
data/             instructions for obtaining datasets (not committed)
docs/             methodology and project screenshots
```

## Key lessons

1. Higher input resolution and a more robust optimization configuration improved the U-Net baseline, but several variables changed together; causal attribution is limited.
2. Tversky loss did not outperform BCE+Dice globally on validation, though the regional trade-off was informative.
3. Model rankings changed between geographic validation and test regions, motivating stricter out-of-region validation.
4. Near-universal positive predictions on Nepal composites exposed an important **cross-domain failure**. Matching tensor shape is not the same as matching SAR data semantics.

## Data, attribution & responsible use

Inspired by training material from the NVIDIA Deep Learning Institute and associated satellite disaster-monitoring case studies. The dataset is related to **Sen1Floods11**; the Nepal assets reference a UNOSAT exercise. Third-party materials are not redistributed in this repository. Refer to each upstream source for licenses, credits and permissible uses. Model weights should only be distributed after checking applicable rights.

This repository intentionally does not claim to estimate flooded area, casualties, or exposed population in Nepal.

## License

**Pending rights review.** Do not add an open-source license or publish third-party assets until redistribution rights have been verified. This notice does not grant reuse rights.
'''


def copied(src, dest, errors):
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    errors.append(str(src))
    return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path.cwd(),help='Existing FloodScope-AI directory')
    p.add_argument('--dest',type=Path,default=None,help='Output publication folder (default: sibling FloodScope-AI-GitHub)')
    p.add_argument('--include-visuals',action='store_true',help='Include original app/assets and 2 error-gallery images ONLY if redistribution rights are confirmed')
    a=p.parse_args()
    source=a.source.resolve()
    dest=(a.dest or source.parent/'FloodScope-AI-GitHub').resolve()
    if source==dest or source in dest.parents or dest in source.parents:
        p.error('Destination must be a separate, non-nested sibling directory; do not overwrite the original.')
    if dest.exists():
        p.error(f'Destination already exists: {dest}; choose a fresh --dest path')
    for required in ['src/dataset.py','app/dashboard.py','app/dopper_theme.css']:
        if not (source/required).exists():
            p.error(f'Expected file missing: {source/required}. Run from your project root.')
    dest.mkdir(parents=True)
    errors=[]
    for sub in ('src','app'):
        for item in (source/sub).rglob('*'):
            if not item.is_file(): continue
            rel=item.relative_to(source)
            if any(part in ['__pycache__','.pytest_cache'] for part in rel.parts):continue
            if sub=='app' and 'assets' in rel.parts and not a.include_visuals:continue
            if item.suffix.lower() not in ('.py','.css','.md','.txt','.toml','.yaml','.yml','.json','.png','.jpg','.jpeg'):continue
            copied(item,dest/rel,errors)
    for file in TOP_LEVEL_FILES:
        if (source/file).exists(): copied(source/file,dest/file,errors)
    for rel in RESULT_FILES:
        if (source/'results'/rel).exists():copied(source/'results'/rel,dest/'results'/rel,errors)
    for folder,files in RESULT_SUBDIRS.items():
        for f in files:
            path=source/'results'/folder/f
            if path.exists():copied(path,dest/'results'/folder/f,errors)
    (dest/'data').mkdir(exist_ok=True)
    (dest/'checkpoints').mkdir(exist_ok=True)
    (dest/'docs').mkdir(exist_ok=True)
    (dest/'docs'/'screenshots').mkdir(exist_ok=True)
    (dest/'.gitignore').write_text(IGNORE,encoding='utf-8')
    (dest/'requirements.txt').write_text(REQUIREMENTS,encoding='utf-8')
    (dest/'README.md').write_text(README,encoding='utf-8')
    (dest/'docs'/'methodology.md').write_text(DOC_METHOD,encoding='utf-8')
    (dest/'data'/'README.md').write_text(DATA_README,encoding='utf-8')
    (dest/'checkpoints'/'README.md').write_text(CKPT_README,encoding='utf-8')
    (dest/'docs'/'screenshots'/'README.md').write_text('Add your own screenshot named dashboard.png here before publishing. Do not use licensed imagery without permission.\n',encoding='utf-8')
    if (source/'docs'/'ARCHIVOS_ORIGINALES.md').exists():
        copied(source/'docs'/'ARCHIVOS_ORIGINALES.md',dest/'docs'/'ARCHIVOS_ORIGINALES.md',errors)
    inventory=[p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()]
    errors_critical=[x for x in errors if x]
    print(f'Prepared {len(inventory)} files at: {dest}')
    print('NOTE: This is a STAGING COPY. Source folder unchanged.')
    print('NOTE: Dataset ZIPs, checkpoints, .venv, Nepal npz are intentionally excluded.')
    print('NOTE: Add a dashboard screenshot, verify dependencies and rights, then publish.')
    if not a.include_visuals:print('Optional: --include-visuals only after checking satellite image redistribution rights.')
    if errors_critical:print('Missing optional source items:',json.dumps(errors_critical,indent=2))

if __name__=='__main__': main()
