import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from math_cnn.images import crop_and_center_drawing, prepare_symbol_image
from writing_cnn.prediction import PredictionPipeline


# Characters

EMNIST_BYCLASS_CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase


# Dataset

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

        if image is None:
            image = Image.new("L", (32, 32), 255)

        if self.augment:
            image = self.augment_image(image)

        image = PredictionPipeline.normalize_stroke_thickness(image)
        tensor = prepare_symbol_image(image, False)

        return tensor, int(example["label"])

    # Augmentation

    @staticmethod
    def augment_image(image):
        image_array = np.asarray(image, dtype=np.uint8)
        height, width = image_array.shape

        if np.random.rand() < 0.65:
            angle = np.random.uniform(-8.0, 8.0)
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            image_array = cv2.warpAffine(image_array, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        if np.random.rand() < 0.65:
            scale = np.random.uniform(0.88, 1.12)
            resized = cv2.resize(image_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            canvas = np.full_like(image_array, 255)

            new_height, new_width = resized.shape
            copy_height, copy_width = min(height, new_height), min(width, new_width)
            y, x = (height - copy_height) // 2, (width - copy_width) // 2
            source_y, source_x = (new_height - copy_height) // 2, (new_width - copy_width) // 2

            canvas[y:y + copy_height, x:x + copy_width] = resized[source_y:source_y + copy_height, source_x:source_x + copy_width]
            image_array = canvas

        if np.random.rand() < 0.6:
            shift_x, shift_y = np.random.randint(-2, 3), np.random.randint(-2, 3)
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image_array = cv2.warpAffine(image_array, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        if np.random.rand() < 0.35:
            shear = np.random.uniform(-0.10, 0.10)
            matrix = np.float32([[1, shear, -shear * height / 2], [0, 1, 0]])
            image_array = cv2.warpAffine(image_array, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        if np.random.rand() < 0.25:
            image_array = EMNISTDataset.elastic_deform(image_array)

        if np.random.rand() < 0.30:
            binary = np.where(image_array < 128, 0, 255).astype(np.uint8)
            kernel = np.ones((np.random.choice([2, 3]),) * 2, dtype=np.uint8)
            operation = np.random.choice(["dilate", "erode"])

            image_array = (
                cv2.dilate(binary, kernel, iterations=1)
                if operation == "dilate"
                else cv2.erode(binary, kernel, iterations=1)
            )

        return Image.fromarray(image_array, mode="L")

    @staticmethod
    def elastic_deform(image, alpha=1.5, sigma=4.0):
        height, width = image.shape

        random_x = np.random.rand(height, width).astype(np.float32) * 2 - 1
        random_y = np.random.rand(height, width).astype(np.float32) * 2 - 1

        dx = cv2.GaussianBlur(random_x, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(random_y, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = x.astype(np.float32) + dx
        map_y = y.astype(np.float32) + dy

        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# EMNIST

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


# Cross-validation

def fold_paths(dataset_root: Path, fold: int) -> tuple[Path, Path]:
    fold_root = dataset_root / "classification-task" / f"fold-{fold}"
    train_csv, test_csv = fold_root / "train.csv", fold_root / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Could not find fold {fold}. Expected {train_csv} and {test_csv}.")

    return train_csv, test_csv