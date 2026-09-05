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
    
def find_word_regions(line: np.ndarray, word_gap_threshold: int = 8) -> list[tuple[int, int]]:
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

def find_conjoined_character_candidates(word: np.ndarray, character_regions: list[tuple[int, int]], width_multiplier: float = 1.8) -> list[tuple[int, int]]:
    if len(character_regions) < 2:
        return []

    widths = np.array([x2 - x1 for x1, x2 in character_regions], dtype=float)
    median_width = np.median(widths)
    
    suspicious = []
    for x1, x2 in character_regions:
        width = x2 - x1

        if width > median_width * width_multiplier:
            suspicious.append((x1, x2))

    return suspicious

def find_conjoined_split(word: np.ndarray, region: tuple[int, int], window: int = 3) -> int | None:
    x1, x2 = region

    projection = np.sum(word[:, x1:x2], axis=0).astype(float)

    if len(projection) < window * 2 + 1:
        return None

    kernel = np.ones(window) / window
    smoothed = np.convolve(projection, kernel, mode="same")

    margin = window
    search = smoothed[margin:-margin]

    if len(search) == 0:
        return None

    split_index = int(np.argmin(search)) + margin

    return x1 + split_index

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

def evaluate_conjoined_split(word: np.ndarray, region: tuple[int, int], split: int, model: torch.nn.Module, device: torch.device) -> tuple[float, int, int]:
    x1, x2 = region

    if split <= x1 or split >= x2:
        return 0.0, -1, -1

    left = word[:, x1:split]
    right = word[:, split:x2]

    left = crop_to_ink(left)
    right = crop_to_ink(right)

    left_image = Image.fromarray((~left * 255).astype(np.uint8), mode="L")
    right_image = Image.fromarray((~right * 255).astype(np.uint8), mode="L")

    left_image = crop_and_center_drawing(left_image)
    right_image = crop_and_center_drawing(right_image)

    if left_image is None or right_image is None:
        return 0.0, -1, -1

    left_prediction, left_confidence = predict_character(model, left_image, device)
    right_prediction, right_confidence = predict_character(model, right_image, device)

    score = left_confidence * right_confidence

    return score, left_prediction, right_prediction

def evaluate_unsplit_character(word: np.ndarray, region: tuple[int, int], model: torch.nn.Module, device: torch.device) -> tuple[float, int]:
    x1, x2 = region

    character = word[:, x1:x2]
    character = crop_to_ink(character)

    image = Image.fromarray((~character * 255).astype(np.uint8), mode="L")
    image = crop_and_center_drawing(image)

    if image is None:
        return 0.0, -1

    prediction, confidence = predict_character(model, image, device)

    return confidence, prediction

def should_split(unsplit_confidence: float, split_score: float, minimum_split_score: float = 0.6, split_margin: float = 1.15) -> bool:
    if split_score < minimum_split_score:
        return False

    return split_score > unsplit_confidence * split_margin

def evaluate_conjoined_split(word: np.ndarray, region: tuple[int, int], split: int, model: torch.nn.Module, device: torch.device) -> tuple[float, int, int]:
    x1, x2 = region

    if split <= x1 or split >= x2:
        return 0.0, -1, -1

    left = crop_to_ink(word[:, x1:split])
    right = crop_to_ink(word[:, split:x2])

    left_image = Image.fromarray((~left * 255).astype(np.uint8), mode="L")
    right_image = Image.fromarray((~right * 255).astype(np.uint8), mode="L")

    left_image = crop_and_center_drawing(left_image)
    right_image = crop_and_center_drawing(right_image)

    if left_image is None or right_image is None:
        return 0.0, -1, -1

    left_prediction, left_confidence = predict_character(model, left_image, device)
    right_prediction, right_confidence = predict_character(model, right_image, device)

    score = left_confidence * right_confidence

    return score, left_prediction, right_prediction
    
    
if __name__ == "__main__":
    from .data import EMNIST_BYCLASS_CHARACTERS

    IMAGE_PATH = "data/test_short.png"
    CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"

    MIN_INK_PIXELS = 8
    MAX_INTERNAL_GAP = 2
    MIN_LINE_HEIGHT = 10
    WORD_GAP_THRESHOLD = 8
    CONJOINED_WIDTH_MULTIPLIER = 1.8

    from pathlib import Path
    import shutil

    results_dir = Path("test_results")

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir()

    # Load image
    image = Image.open(IMAGE_PATH).convert("L")

    print(f"Original image size: {image.size}")

    # Binary image
    binary = convert_image_binary(image)

    print(f"Binary shape: {binary.shape}")
    print(f"Handwriting pixels: {np.sum(binary)}")
    print(f"Foreground percentage: {100 * np.mean(binary):.2f}%")

    binary_image = Image.fromarray((~binary * 255).astype(np.uint8), mode="L")
    binary_image.save(results_dir / "binary_test.png")

    # Writing region
    writing = find_writing_boundaries(binary)

    print(f"Writing region shape: {writing.shape}")

    writing_image = Image.fromarray((~writing * 255).astype(np.uint8), mode="L")
    writing_image.save(results_dir / "writing_test.png")

    # Line segmentation
    line_regions = find_line_regions(writing, min_ink_pixels=MIN_INK_PIXELS, max_internal_gap=MAX_INTERNAL_GAP, min_line_height=MIN_LINE_HEIGHT)
    line_regions = filter_line_regions(line_regions, min_height=MIN_LINE_HEIGHT)

    print(f"\nNumber of lines: {len(line_regions)}")

    lines = crop_lines(writing, line_regions)

    for line_index, line in enumerate(lines):
        print(f"Line {line_index}: shape={line.shape}")

        line_dir = results_dir / f"line_{line_index}"
        line_dir.mkdir()

        line_image = Image.fromarray((~line * 255).astype(np.uint8), mode="L")
        line_image.save(line_dir / "line.png")

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\nDevice: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model = WritingCNN(checkpoint["number_of_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    class_names = list(EMNIST_BYCLASS_CHARACTERS)

    # Word and character segmentation
    for line_index, line in enumerate(lines):
        print(f"\nLINE {line_index}")

        line_dir = results_dir / f"line_{line_index}"

        word_regions = find_word_regions(line, word_gap_threshold=WORD_GAP_THRESHOLD)

        print(f"Number of words: {len(word_regions)}")

        for word_index, (x1, x2) in enumerate(word_regions):
            word_dir = line_dir / f"word_{word_index}"
            word_dir.mkdir()

            word = line[:, x1:x2]
            word = crop_to_ink(word)

            word_image = Image.fromarray((~word * 255).astype(np.uint8), mode="L")
            word_image.save(word_dir / "word.png")

            print(f"\nWord {word_index}: shape={word.shape}")

            character_regions = find_character_regions(word)

            print(f"Initial character regions: {len(character_regions)}")

            # Conjoined character detection
            conjoined_candidates = find_conjoined_character_candidates(word, character_regions, width_multiplier=CONJOINED_WIDTH_MULTIPLIER)

            print(f"Conjoined candidates: {len(conjoined_candidates)}")

            for candidate in conjoined_candidates:
                split = find_conjoined_split(word, candidate)

                print(f"  Candidate {candidate} -> proposed split: {split}")

                if split is None:
                    continue

                split_score, left_prediction, right_prediction = evaluate_conjoined_split(word, candidate, split, model, device)

                print(f"    Split score: {split_score:.3f}")

                if 0 <= left_prediction < len(class_names):
                    print(f"    Left prediction: {class_names[left_prediction]}")

                if 0 <= right_prediction < len(class_names):
                    print(f"    Right prediction: {class_names[right_prediction]}")

            # Character crops and predictions
            characters = crop_characters(word, character_regions)

            print(f"Saving {len(characters)} character crops")

            for character_index, character in enumerate(characters):
                character.save(word_dir / f"character_{character_index}.png")

                prediction, confidence = predict_character(model, character, device)

                predicted_character = class_names[prediction] if 0 <= prediction < len(class_names) else "?"

                print(f"  Character {character_index}: {predicted_character} (confidence={confidence:.3f})")