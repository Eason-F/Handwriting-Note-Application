import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from math_cnn.images import prepare_symbol_image, crop_and_center_drawing


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
        image = crop_and_center_drawing(image)

        if self.augment:
            image = self.augment_image(image)

        tensor = prepare_symbol_image(image, False)
        label = int(example["label"])

        return tensor, label

    @staticmethod
    def augment_image(image: Image.Image) -> Image.Image:
        image_array = np.asarray(image).astype(np.uint8)

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-8.0, 8.0)
            height, width = image_array.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            image_array = cv2.warpAffine(
                image_array,
                matrix,
                (width, height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

        if np.random.rand() < 0.5:
            scale_x = np.random.uniform(0.85, 1.15)
            scale_y = np.random.uniform(0.85, 1.15)

            height, width = image_array.shape
            resized = cv2.resize(
                image_array,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_LINEAR,
            )

            canvas = np.full((height, width), 255, dtype=np.uint8)

            new_height, new_width = resized.shape
            copy_height = min(new_height, height)
            copy_width = min(new_width, width)

            y = max((height - copy_height) // 2, 0)
            x = max((width - copy_width) // 2, 0)

            source_y = max((new_height - copy_height) // 2, 0)
            source_x = max((new_width - copy_width) // 2, 0)

            canvas[
                y:y + copy_height,
                x:x + copy_width,
            ] = resized[
                source_y:source_y + copy_height,
                source_x:source_x + copy_width,
            ]

            image_array = canvas

        if np.random.rand() < 0.5:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)

            matrix = np.float32([
                [1, 0, shift_x],
                [0, 1, shift_y],
            ])

            image_array = cv2.warpAffine(
                image_array,
                matrix,
                (image_array.shape[1], image_array.shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

        if np.random.rand() < 0.5:
            shear = np.random.uniform(-0.15, 0.15)
            height, width = image_array.shape

            matrix = np.float32([
                [1, shear, -shear * height / 2],
                [0, 1, 0],
            ])

            image_array = cv2.warpAffine(
                image_array,
                matrix,
                (width, height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

        if np.random.rand() < 0.5:
            operation = np.random.choice(["dilate", "erode"])

            binary = np.where(image_array < 128, 0, 255).astype(np.uint8)

            kernel = np.ones((3, 3), dtype=np.uint8)

            if operation == "dilate":
                image_array = cv2.dilate(binary, kernel, iterations=1)
            else:
                image_array = cv2.erode(binary, kernel, iterations=1)

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