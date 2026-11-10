# Checkpoints

Model checkpoints are intentionally excluded from source control.

Expected filenames for the interactive model explorer:

- `v2_512_best.pt` — U-Net/ResNet-18, 512×512.
- `v4_deeplab_best.pt` — DeepLabV3+/ResNet-18, 512×512.

The training code is included. Subject to rights review, trained weights may be provided as separate GitHub Release assets. Until then the dashboard can display documented metrics without running weight-dependent inference.

Note that weights derived from third-party data and training materials may have redistribution conditions. Verify them before releasing.
