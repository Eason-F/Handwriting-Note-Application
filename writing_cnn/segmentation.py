import numpy as np
from PIL import Image, ImageOps
import torch

from math_cnn.images import crop_and_center_drawing, prepare_symbol_image
from .model import WritingCNN

def convert_image_binary(image: Image.Image, threshold = 200) -> np.ndarray:
    pixels = np.array(image.convert("L"))
    return pixels < threshold

def find_writing_boundaries(binary: np.ndarray) -> np.ndarray:
    '''Crop canvas to smallest '''
    y_pt, x_pt = np.where(binary)
    if len(x_pt) == 0:
        raise ValueError("No handwriting detected.")
    
    return binary[
        y_pt.min():y_pt.max() + 1,
        x_pt.min():x_pt.max() + 1,
    ]
    
def crop_to_ink(image: np.ndarray) -> np.ndarray:
    ys, xs = np.where(image)

    if len(xs) == 0:
        raise ValueError("No handwriting detected.")

    x1 = xs.min()
    x2 = xs.max() + 1
    y1 = ys.min()
    y2 = ys.max() + 1

    return image[y1:y2, x1:x2]
    
def find_line_regions(binary: np.ndarray, min_ink_pixels: int = 8, max_internal_gap: int = 2, min_line_height: int = 10) -> list[tuple[int, int]]:
    horizontal_projection = np.sum(binary, axis=1)

    active = horizontal_projection >= min_ink_pixels

    regions = []
    start = None
    gap_start = None

    for y, has_ink in enumerate(active):
        if has_ink:
            if start is None:
                start = y

            gap_start = None

        elif start is None:
            continue
        
        if gap_start is None:
            gap_start = y

        gap_width = y - gap_start

        if gap_width > max_internal_gap:
            end = gap_start

            if end - start >= min_line_height:
                regions.append((start, end))

            start = None
            gap_start = None

    if start is not None:
        end = gap_start if gap_start is not None else len(active)

        if end - start >= min_line_height:
            regions.append((start, end))

    return regions

def filter_line_regions(regions: list[tuple[int, int]], min_height: int = 5) -> list[tuple[int, int]]:
    return [(y1, y2) for y1, y2 in regions if y2 - y1 >= min_height]

def crop_lines(binary: np.ndarray, line_regions: list[tuple[int, int]]) -> list[np.ndarray]:
    lines = []

    for y1, y2 in line_regions:
        line = binary[y1:y2, :]
        line = crop_to_ink(line)
        lines.append(line)

    return lines
    
def find_gap_widths(line: np.ndarray) -> list[tuple[int, int, int]]:
    projection = np.sum(line, axis=0)
    active = projection > 0

    gaps = []
    start = None

    for x, has_ink in enumerate(active):
        if not has_ink and start is None:
            start = x
        elif has_ink and start is not None:
            width = x - start
            gaps.append((start, x, width))
            start = None

    if start is not None:
        width = len(active) - start
        gaps.append((start, len(active), width))

    return gaps
    
def find_word_regions(line: np.ndarray, word_gap_threshold: int = 6) -> list[tuple[int, int]]:
    projection = np.sum(line, axis=0)
    active = projection > 0

    word_regions = []
    word_start = None
    gap_start = None

    for x, has_ink in enumerate(active):
        if has_ink:
            if word_start is None:
                word_start = x

            gap_start = None

        elif word_start is not None:
            if gap_start is None:
                gap_start = x

            gap_width = x - gap_start

            if gap_width > word_gap_threshold:
                word_regions.append((word_start, gap_start))
                word_start = None
                gap_start = None

    if word_start is not None:
        word_regions.append((word_start, len(active)))

    return word_regions
    
    
def find_character_regions(word: np.ndarray) -> list[tuple[int, int]]:
    projection = np.sum(word, axis=0)
    active = projection > 0

    regions = []
    start = None

    for x, has_ink in enumerate(active):
        if has_ink and start is None:
            start = x
        elif not has_ink and start is not None:
            regions.append((start, x))
            start = None

    if start is not None:
        regions.append((start, len(active)))

    return regions

def crop_characters(word: np.ndarray, character_regions: list[tuple[int, int]]) -> list[Image.Image]:
    characters = []

    for x1, x2 in character_regions:
        character = word[:, x1:x2]

        character_image = Image.fromarray((~character * 255).astype(np.uint8), mode="L")
        character_image = crop_and_center_drawing(character_image)

        if character_image is not None:
            characters.append(character_image)

    return characters

def predict_character(model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
    tensor = prepare_symbol_image(character)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    return prediction.item(), confidence.item()
    
if __name__ == "__main__":
    from .data import EMNIST_BYCLASS_CHARACTERS
    
    image = Image.open("data/test_short.png")
    binary = convert_image_binary(image)
    binary_image = Image.fromarray((~binary * 255).astype(np.uint8), mode="L")
    binary_image.save("binary_test.png")
    writing = find_writing_boundaries(binary)
    print("Image shape:", binary.shape)
    print("Handwriting pixels:", np.sum(binary))
    print("Percentage:", 100 * np.mean(binary))
    
    line_regions = find_line_regions(writing, min_ink_pixels=8, max_internal_gap=2, min_line_height=10)
    line_regions = filter_line_regions(line_regions, min_height=10)
    lines = crop_lines(binary, line_regions)
    
    print("Number of lines:", len(line_regions))
    for i, (y1, y2) in enumerate(line_regions):
        print(f"Line {i}: y={y1}..{y2}, height={y2 - y1}, ")
        
    for i, line in enumerate(lines):
        print(f"Line {i}: shape={line.shape}")
    
    line = lines[0]
    word_regions = find_word_regions(line, word_gap_threshold=8)
    print(word_regions)
    
    gaps = find_gap_widths(lines[0])

    words = []
    for x1, x2 in word_regions:
        word = line[:, x1:x2]
        word = crop_to_ink(word)
        words.append(word)
        
    word = words[1]
    character_regions = find_character_regions(word)
    characters = crop_characters(word, character_regions)

    for i, character in enumerate(characters):
        character.save(f"{i}.png")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load("checkpoints/writing_cnn.pt", map_location=device, weights_only=True)
    model = WritingCNN(checkpoint["number_of_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    device = device
    
    for character_image in characters:
        prediction, confidence = predict_character(model, character_image, device)
        character = list(EMNIST_BYCLASS_CHARACTERS)[prediction]

        print(character, confidence)