"""HiRISE 227×227 grayscale crop dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class HiRISEDataset(Dataset):
    """
    Loads competition crops. If crop_meta_csv is given, order and source_image_id
    follow that table; otherwise we scan *.jpg in image_dir.
    """

    def __init__(
        self,
        image_dir: str | Path,
        crop_meta_csv: str | Path | None = None,
        preload: bool = True,
    ):
        self.image_dir = Path(image_dir)
        if crop_meta_csv is not None:
            meta = pd.read_csv(crop_meta_csv)
            self.filenames = meta["filename"].tolist()
            self.source_ids = meta["source_image_id"].tolist()
        else:
            self.filenames = sorted(p.name for p in self.image_dir.glob("*.jpg"))
            self.source_ids = ["UNK"] * len(self.filenames)

        self.images = None
        if preload:
            # Keep float32 stack in RAM so training is not JPEG-bound.
            arrs = []
            for name in tqdm(self.filenames, desc="Preload images", leave=False):
                img = Image.open(self.image_dir / name).convert("L")
                arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
            self.images = np.stack(arrs, axis=0)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        if self.images is not None:
            arr = self.images[idx]
        else:
            img = Image.open(self.image_dir / name).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0)
        return tensor, name, self.source_ids[idx]
