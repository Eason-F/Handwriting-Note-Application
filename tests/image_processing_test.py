from pathlib import Path
import shutil

import torch
from PIL import Image

from writing_cnn.data import EMNISTDataset, load_emnist_source


# Configuration

DATA_DIR = Path("data/tfds/parquet")
RESULTS_DIR = Path("test_results")
NUM_IMAGES = 100


# Clear previous test results

if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir(parents=True)


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

    output_path = (
        RESULTS_DIR
        / f"{index:03d}_label_{label}.png"
    )

    image.save(output_path)


print(
    f"Saved {num_images} processed images "
    f"to {RESULTS_DIR}/"
)