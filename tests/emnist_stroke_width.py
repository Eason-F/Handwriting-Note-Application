from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from writing_cnn.data import EMNISTDataset, load_emnist_source
from writing_cnn.prediction import PredictionPipeline


# Configuration

DATA_DIR = Path("data/tfds/parquet")
NUM_IMAGES = 1000



# Load EMNIST test data

print("Loading EMNIST test dataset...")

emnist_source = load_emnist_source(
    DATA_DIR,
    "test",
)

dataset = EMNISTDataset(
    emnist_source,
)


# Save processed images

print(
    f"Saving the first {NUM_IMAGES} processed "
    f"EMNIST test images..."
)

num_images = min(
    NUM_IMAGES,
    len(dataset),
)

widths = []

for index in range(num_images):
    tensor, _ = dataset[index]
    image = ((tensor.squeeze() * 127.5) + 127.5).clamp(0, 255).byte().numpy()
    width = PredictionPipeline.estimate_stroke_width(image < 128)

    if width > 0:
        widths.append(width)

print(
    f"median={np.median(widths):.2f}px "
    f"mean={np.mean(widths):.2f}px "
    f"std_dev={np.std(widths):.2f}px"
)
    
for width in range(1, 8):
    image = np.full((32, 32), 255, np.uint8)
    cv2.line(image, (4, 4), (27, 27), 0, width)

    measured = PredictionPipeline.estimate_stroke_width(image < 128)

    print(f"actual={width}px measured={measured:.2f}px")

    