from pathlib import Path

import torch
import numpy as np
from PIL import Image

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

stroke_sums = []
for index in range(num_images):
    tensor, label = dataset[index]

    image = tensor.squeeze(0).detach().cpu()

    image = (
        image.clamp(0, 1) * 255
    ).to(torch.uint8)

    image = Image.fromarray(
        image.numpy(),
        mode="L",
    )
    
    image_array = np.asarray(image.convert("L"))
    ink = image_array < 128
    
    stroke_sums.append(PredictionPipeline.estimate_stroke_width(ink))
    
array = np.array(stroke_sums)
ave = np.average(array)
std_dev = np.std(array)

print(f"Average stroke width {ave}")
print(f"Deviation of {std_dev}")

    