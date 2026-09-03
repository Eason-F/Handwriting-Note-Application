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


def crop_and_center_drawing(
    image: Image.Image,
    margin_ratio: float = 0.25,
) -> Optional[Image.Image]:
    grayscale = image.convert("L")
    ink = ImageOps.invert(grayscale)
    bounding_box = ink.point(lambda value: 255 if value > 20 else 0).getbbox()
    if bounding_box is None:
        return None

    cropped = grayscale.crop(bounding_box)
    
    # 1. Add proportional margins to height and width independently
    pad_w = max(4, round(cropped.width * margin_ratio))
    pad_h = max(4, round(cropped.height * margin_ratio))
    
    padded_width = cropped.width + 2 * pad_w
    padded_height = cropped.height + 2 * pad_h
    
    # 2. Make the final canvas a perfect square based on the largest padded side
    canvas_side = max(padded_width, padded_height)
    
    # 3. Create white canvas and paste the cropped image perfectly centered
    centered = Image.new("L", (canvas_side, canvas_side), color=255)
    position = (
        (canvas_side - cropped.width) // 2,
        (canvas_side - cropped.height) // 2,
    )
    centered.paste(cropped, position)
    
    return centered.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


