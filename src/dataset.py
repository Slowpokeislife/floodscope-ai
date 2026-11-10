"""Load Sen1Floods11 training PNG image/mask pairs from ZIP without extraction."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_PREFIX = 'images/all_images/'
MASK_PREFIX = 'masks/all_masks/'


def discover(zip_path):
    with ZipFile(zip_path) as z:
        names = set(z.namelist())
    images = {Path(n).stem: n for n in names if n.startswith(IMG_PREFIX) and n.lower().endswith('.png')}
    masks = {Path(n).stem: n for n in names if n.startswith(MASK_PREFIX) and n.lower().endswith('.png')}
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise ValueError('No matching images/masks in data/flood_data.zip')
    return common


def region_of(sample_id):
    return sample_id.rsplit('_', 1)[0]


def create_splits(ids, seed=42, method='region'):
    rng = random.Random(seed)
    if method == 'region':
        groups = sorted({region_of(i) for i in ids})
        if len(groups) < 5:
            raise ValueError('Need at least 5 regions for group-wise split')
        rng.shuffle(groups)
        n_test = max(1, round(0.15 * len(groups)))
        n_val = max(1, round(0.15 * len(groups)))
        test_regions = set(groups[:n_test])
        val_regions = set(groups[n_test:n_test + n_val])
        train_regions = set(groups[n_test + n_val:])
        return {
            'train': [i for i in ids if region_of(i) in train_regions],
            'val': [i for i in ids if region_of(i) in val_regions],
            'test': [i for i in ids if region_of(i) in test_regions],
        }
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_test = max(1, round(0.15 * len(ids)))
    n_val = max(1, round(0.15 * len(ids)))
    return {'train': sorted(shuffled[n_test + n_val:]),
            'val': sorted(shuffled[n_test:n_test + n_val]),
            'test': sorted(shuffled[:n_test])}


class FloodZipDataset(Dataset):
    def __init__(self, zip_path, ids, size=256, augment=False):
        self.zip_path = str(zip_path)
        self.ids = list(ids)
        self.size = size
        self.augment = augment
        self._zip = None

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip'] = None
        return state

    def __getitem__(self, idx):
        if self._zip is None:
            self._zip = ZipFile(self.zip_path)
        sample_id = self.ids[idx]
        image_path = IMG_PREFIX + sample_id + '.png'
        mask_path = MASK_PREFIX + sample_id + '.png'
        with self._zip.open(image_path) as f:
            img = Image.open(BytesIO(f.read())).convert('RGB')
        with self._zip.open(mask_path) as f:
            mask = Image.open(BytesIO(f.read())).convert('L')
        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.uint8) == 255).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                x, y = np.flip(x, axis=1), np.flip(y, axis=1)
            if random.random() < 0.5:
                x, y = np.flip(x, axis=0), np.flip(y, axis=0)
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        y = np.ascontiguousarray(y[None, :, :])
        return torch.from_numpy(x), torch.from_numpy(y), sample_id
