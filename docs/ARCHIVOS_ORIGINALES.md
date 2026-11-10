# Which original files to copy

## Required

Copy **only** `data/flood_data.zip` from your original course directory into this new repository's `data/flood_data.zip`. Leave it compressed.

## Keep separately (not needed for v1)

- `data/UNOSAT_case_study.zip`: keep in the original course directory, not copied to v1.
- `tao_project/unet/export/sample_resnet18.engine`: keep as historical NVIDIA artifact, not required for PyTorch training/inference. It is not interchangeable with the PyTorch checkpoint.
- `tao_project/spec_files/resnet18/{combined_config,dataset_config,model_config,training_config}.txt`: retain for reference; they contain `FIXME` placeholders and are not v1 configurations.
- `00_introduction.ipynb` through `04_unosat_flood_event_case_study.ipynb`: source course material; keep locally and cite origin, do not copy to the public repository.
- `images/`: course images/assets; do not copy to public repository.
- `ngc_assets/`: bundled NVIDIA NGC CLI for Linux; unnecessary for the new Python pipeline.
- `models/`: empty; not used.
- `full_vv_image.png`, `full_vh_image.png`: optional original artifacts; keep locally until source/rights are checked.

## GitHub

`.gitignore` excludes `data/*`, models and output artifacts. Do not force-add datasets, third-party images or course notebooks without checking their licenses.
