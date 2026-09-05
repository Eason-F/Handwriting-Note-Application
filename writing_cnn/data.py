import csv
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
import matplotlib._mathtext_data as mtd

from math_cnn.images import prepare_symbol_image

EMNIST_BYCLASS_CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase
class EMNISTDataset(Dataset):
    def __init__(self, tfds_source):
        self.source = tfds_source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        example = self.source[index]
        image_array = np.asarray(example["image"]).squeeze().T
        image = ImageOps.invert(Image.fromarray(image_array.astype(np.uint8), mode="L"))
        label = int(example["label"])
        tensor = prepare_symbol_image(image, True)
        return tensor, label


def load_emnist_source(data_dir: Path, split: str):
    """Download if needed and return a TensorFlow-free TFDS data source."""
    import tensorflow_datasets as tfds

    return tfds.data_source(
        "emnist/byclass",
        split=split,
        data_dir=data_dir,
        download=True,
        builder_kwargs={"file_format": "parquet"},
        download_and_prepare_kwargs={"download_dir": data_dir.parent / "downloads"},
    )


def fold_paths(dataset_root: Path, fold: int) -> tuple[Path, Path]:
    fold_root = dataset_root / "classification-task" / f"fold-{fold}"
    train_csv = fold_root / "train.csv"
    test_csv = fold_root / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Could not find fold {fold}. Expected {train_csv} and {test_csv}."
        )
    return train_csv, test_csv