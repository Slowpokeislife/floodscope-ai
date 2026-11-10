# FloodScope AI 

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
.venv\Scripts\Activate.ps1
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
