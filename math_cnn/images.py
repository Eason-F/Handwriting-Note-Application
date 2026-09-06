import random
from typing import Optional

import torch
from PIL import Image, ImageOps


IMAGE_SIZE = 32


def prepare_symbol_image(image: Image.Image, augment: bool = False) -> torch.Tensor:
    """Convert a PIL image to the normalized tensor used by the model."""
    image = image.convert("L")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    if augment:
        image = image.rotate(
            random.uniform(-7.0, 7.0),
            resample=Image.Resampling.BILINEAR,
            translate=(random.randint(-2, 2), random.randint(-2, 2)),
            fillcolor=255,
        )

    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE).to(torch.float32) / 255.0
    return (tensor - 0.5) / 0.5


def crop_and_center_drawing(image: Image.Image, margin_ratio: float = 0.20) -> Optional[Image.Image]:
    grayscale = image.convert("L")
    ink = ImageOps.invert(grayscale)
    bounding_box = ink.point(lambda value: 255 if value > 20 else 0).getbbox()

    if bounding_box is None:
        return None

    cropped = grayscale.crop(bounding_box)

    target_height = round(IMAGE_SIZE * (1.0 - 2.0 * margin_ratio))
    target_width = round(IMAGE_SIZE * (1.0 - 2.0 * margin_ratio))

    scale = min(
        target_height / cropped.height,
        target_width / cropped.width,
    )

    new_width = max(1, round(cropped.width * scale))
    new_height = max(1, round(cropped.height * scale))

    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)

    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2

    canvas.paste(resized, (x, y))

    return canvas


