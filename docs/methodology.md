# Methodology and evaluation protocol

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
