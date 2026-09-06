import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from math_cnn.images import prepare_symbol_image


EMNIST_BYCLASS_CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase


class EMNISTDataset(Dataset):
    def __init__(self, tfds_source, augment=False):
        self.source = tfds_source
        self.augment = augment

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        example = self.source[index]
        image_array = np.asarray(example["image"]).squeeze().T
        image = ImageOps.invert(Image.fromarray(image_array.astype(np.uint8), mode="L"))

        if self.augment:
            image = self.augment_image(image)

        tensor = prepare_symbol_image(image, True)
        label = int(example["label"])

        return tensor, label

    @staticmethod
    def augment_image(image: Image.Image) -> Image.Image:
        image_array = np.asarray(image)

        operation = np.random.choice(["none", "dilate", "erode"])

        if operation == "dilate":
            binary = np.where(image_array < 128, 0, 255).astype(np.uint8)
            image_array = cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)

        elif operation == "erode":
            binary = np.where(image_array < 128, 0, 255).astype(np.uint8)
            image_array = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)

        return Image.fromarray(image_array)


def load_emnist_source(data_dir: Path, split: str):
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
        raise FileNotFoundError(f"Could not find fold {fold}. Expected {train_csv} and {test_csv}.")

    return train_csv, test_csv