# Dataset acquisition

`flood_data.zip` and `UNOSAT_case_study.zip` are **not included** in this repository.

This research originated from an NVIDIA Deep Learning Institute disaster-risk exercise involving Sentinel-1 imagery and the Sen1Floods11 dataset. Obtain data from an authorized source and review applicable licenses/terms before reuse. For the existing scripts, place the authorized training archive at `data/flood_data.zip`. The data interface expects `images/all_images/*.png` and `masks/all_masks/*.png` paired by stem within that ZIP.

The UNOSAT/Nepal ZIP is an *exploratory domain-shift investigation only*: it is not necessary to reproduce labeled segmentation metrics. There is no validated Nepal ground truth in this project.

Do not interpret model probability maps as operational flood determinations.
