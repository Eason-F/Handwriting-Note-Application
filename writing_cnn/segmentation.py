from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import torch

from math_cnn.images import crop_and_center_drawing, prepare_symbol_image
from .model import WritingCNN


class LineSegmenter:
    def __init__(self, min_ink_pixels: int = 8, max_internal_gap: int = 2, min_line_height: int = 10) -> None:
        self.min_ink_pixels = min_ink_pixels
        self.max_internal_gap = max_internal_gap
        self.min_line_height = min_line_height

    def find_regions(self, binary: np.ndarray) -> list[tuple[int, int]]:
        projection = np.sum(binary, axis=1)
        active = projection >= self.min_ink_pixels

        regions = []
        start = None
        gap = 0

        for y, ink in enumerate(active):
            if ink:
                if start is None:
                    start = y
                gap = 0
                continue

            if start is None:
                continue

            gap += 1

            if gap <= self.max_internal_gap:
                continue

            end = y - gap

            if end - start >= self.min_line_height:
                regions.append((start, end))

            start = None
            gap = 0

        if start is not None:
            end = len(active) - gap
            if end - start >= self.min_line_height:
                regions.append((start, end))

        return regions

    def crop(self, binary: np.ndarray, regions: list[tuple[int, int]]) -> list[np.ndarray]:
        return [self.crop_to_ink(binary[y1:y2, :]) for y1, y2 in regions]

    @staticmethod
    def crop_to_ink(image: np.ndarray) -> np.ndarray:
        ys, xs = np.where(image)

        if len(xs) == 0:
            raise ValueError("No handwriting detected.")

        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class WordSegmenter:
    def __init__(self, word_gap_threshold: int = 8) -> None:
        self.word_gap_threshold = word_gap_threshold

    def find_regions(self, line: np.ndarray) -> list[tuple[int, int]]:
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

                if gap_width > self.word_gap_threshold:
                    word_regions.append((word_start, gap_start))
                    word_start = None
                    gap_start = None

        if word_start is not None:
            word_regions.append((word_start, len(active)))

        return word_regions

    def crop(self, line: np.ndarray, regions: list[tuple[int, int]]) -> list[np.ndarray]:
        words = []

        for x1, x2 in regions:
            word = line[:, x1:x2]
            word = self.crop_to_ink(word)
            words.append(word)

        return words

    @staticmethod
    def crop_to_ink(image: np.ndarray) -> np.ndarray:
        ys, xs = np.where(image)

        if len(xs) == 0:
            raise ValueError("No handwriting detected.")

        x1 = xs.min()
        x2 = xs.max() + 1
        y1 = ys.min()
        y2 = ys.max() + 1

        return image[y1:y2, x1:x2]


class CharacterSegmenter:
    def find_regions(self, word: np.ndarray) -> list[tuple[int, int]]:
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

    def crop(self, word: np.ndarray, regions: list[tuple[int, int]]) -> list[Image.Image]:
        characters = []

        for x1, x2 in regions:
            character = word[:, x1:x2]

            character_image = Image.fromarray((~character * 255).astype(np.uint8), mode="L")
            character_image = crop_and_center_drawing(character_image)

            if character_image is not None:
                characters.append(character_image)

        return characters


class ConjoinedCharacterSegmenter:
    def __init__(self, width_multiplier: float = 1.8, min_character_width: int = 5) -> None:
        self.width_multiplier = width_multiplier
        self.min_character_width = min_character_width

    def find_candidates(self, word: np.ndarray, character_regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(character_regions) < 2:
            return []

        widths = np.array([x2 - x1 for x1, x2 in character_regions], dtype=float)
        median_width = np.median(widths)

        candidates = []

        for x1, x2 in character_regions:
            width = x2 - x1

            if width > median_width * self.width_multiplier:
                candidates.append((x1, x2))

        return candidates

    def find_split(self, word: np.ndarray, region: tuple[int, int]) -> int | None:
        x1, x2 = region

        if x2 - x1 < self.min_character_width * 2:
            return None

        projection = np.sum(word[:, x1:x2], axis=0).astype(float)

        if projection.max() == 0:
            return None

        split_index = int(np.argmin(projection))
        split = x1 + split_index

        left_width = split - x1
        right_width = x2 - split

        if left_width < self.min_character_width or right_width < self.min_character_width:
            return None

        return split

    def evaluate_split(self, word: np.ndarray, region: tuple[int, int], split: int, model: torch.nn.Module, device: torch.device) -> tuple[float, int, int]:
        x1, x2 = region

        if split <= x1 or split >= x2:
            return 0.0, -1, -1

        left = word[:, x1:split]
        right = word[:, split:x2]

        left = self.crop_to_ink(left)
        right = self.crop_to_ink(right)

        left_image = Image.fromarray((~left * 255).astype(np.uint8), mode="L")
        right_image = Image.fromarray((~right * 255).astype(np.uint8), mode="L")

        left_image = crop_and_center_drawing(left_image)
        right_image = crop_and_center_drawing(right_image)

        if left_image is None or right_image is None:
            return 0.0, -1, -1

        left_prediction, left_confidence = self.predict(model, left_image, device)
        right_prediction, right_confidence = self.predict(model, right_image, device)

        score = np.sqrt(left_confidence * right_confidence)

        return score, left_prediction, right_prediction

    def evaluate_unsplit(self, word: np.ndarray, region: tuple[int, int], model: torch.nn.Module, device: torch.device) -> tuple[float, int]:
        x1, x2 = region

        character = word[:, x1:x2]
        character = self.crop_to_ink(character)

        image = Image.fromarray((~character * 255).astype(np.uint8), mode="L")
        image = crop_and_center_drawing(image)

        if image is None:
            return 0.0, -1

        prediction, confidence = self.predict(model, image, device)

        return confidence, prediction

    def should_split(self, unsplit_confidence: float, split_score: float, minimum_split_score: float = 0.6, split_margin: float = 1.15) -> bool:
        if split_score < minimum_split_score:
            return False

        return split_score > unsplit_confidence * split_margin

    def process(self, word: np.ndarray, character_regions: list[tuple[int, int]], model: torch.nn.Module, device: torch.device) -> list[tuple[int, int]]:
        regions = character_regions.copy()

        candidates = self.find_candidates(word, regions)

        for candidate in candidates:
            if candidate not in regions:
                continue

            split = self.find_split(word, candidate)

            if split is None:
                continue

            unsplit_confidence, _ = self.evaluate_unsplit(word, candidate, model, device)
            split_score, _, _ = self.evaluate_split(word, candidate, split, model, device)

            if self.should_split(unsplit_confidence, split_score):
                index = regions.index(candidate)

                regions[index:index + 1] = [
                    (candidate[0], split),
                    (split, candidate[1]),
                ]

        return regions

    @staticmethod
    def crop_to_ink(image: np.ndarray) -> np.ndarray:
        ys, xs = np.where(image)

        if len(xs) == 0:
            raise ValueError("No handwriting detected.")

        x1 = xs.min()
        x2 = xs.max() + 1
        y1 = ys.min()
        y2 = ys.max() + 1

        return image[y1:y2, x1:x2]

    @staticmethod
    def predict(model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
        tensor = prepare_symbol_image(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        return prediction.item(), confidence.item()


class SegmentationPipeline:
    def __init__(self, line_segmenter: LineSegmenter, word_segmenter: WordSegmenter, character_segmenter: CharacterSegmenter, conjoined_segmenter: ConjoinedCharacterSegmenter | None = None) -> None:
        self.line_segmenter = line_segmenter
        self.word_segmenter = word_segmenter
        self.character_segmenter = character_segmenter
        self.conjoined_segmenter = conjoined_segmenter

    @staticmethod
    def binarize(image: Image.Image, threshold: int = 200) -> np.ndarray:
        pixels = np.array(image.convert("L"))
        return pixels < threshold

    @staticmethod
    def find_writing_region(binary: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary)

        if len(xs) == 0:
            raise ValueError("No handwriting detected.")

        return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def segment(self, image: Image.Image, model: torch.nn.Module | None = None, device: torch.device | None = None) -> list[list[list[Image.Image]]]:
        binary = self.binarize(image)
        writing = self.find_writing_region(binary)

        line_regions = self.line_segmenter.find_regions(writing)
        lines = self.line_segmenter.crop(writing, line_regions)

        result = []

        for line in lines:
            word_regions = self.word_segmenter.find_regions(line)
            words = self.word_segmenter.crop(line, word_regions)

            line_result = []

            for word in words:
                character_regions = self.character_segmenter.find_regions(word)

                if self.conjoined_segmenter is not None:
                    if model is None or device is None:
                        raise ValueError("Model and device are required for conjoined-character processing.")

                    character_regions = self.conjoined_segmenter.process(word, character_regions, model, device)

                characters = self.character_segmenter.crop(word, character_regions)
                line_result.append(characters)

            result.append(line_result)

        return result
    
if __name__ == "__main__":
    from .data import EMNIST_BYCLASS_CHARACTERS

    IMAGE_PATH = "data/test_short.png"
    CHECKPOINT_PATH = "checkpoints/writing_cnn.pt"

    results_dir = Path("test_results")

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir()

    line_segmenter = LineSegmenter(
        min_ink_pixels=8,
        max_internal_gap=2,
        min_line_height=10,
    )

    word_segmenter = WordSegmenter(
        word_gap_threshold=8,
    )

    character_segmenter = CharacterSegmenter()

    conjoined_segmenter = ConjoinedCharacterSegmenter(
        width_multiplier=1.8,
        min_character_width=5,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model = WritingCNN(checkpoint["number_of_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    class_names = list(EMNIST_BYCLASS_CHARACTERS)

    image = Image.open(IMAGE_PATH).convert("L")

    print(f"Original image size: {image.size}")

    binary = SegmentationPipeline.binarize(image)

    print(f"Binary shape: {binary.shape}")
    print(f"Handwriting pixels: {np.sum(binary)}")
    print(f"Foreground percentage: {100 * np.mean(binary):.2f}%")

    binary_image = Image.fromarray((~binary * 255).astype(np.uint8), mode="L")
    binary_image.save(results_dir / "binary_test.png")

    writing = SegmentationPipeline.find_writing_region(binary)

    print(f"Writing region shape: {writing.shape}")

    writing_image = Image.fromarray((~writing * 255).astype(np.uint8), mode="L")
    writing_image.save(results_dir / "writing_test.png")

    line_regions = line_segmenter.find_regions(writing)
    lines = line_segmenter.crop(writing, line_regions)

    print(f"\nNumber of lines: {len(lines)}")

    for line_index, line in enumerate(lines):
        print(f"Line {line_index}: shape={line.shape}")

        line_dir = results_dir / f"line_{line_index}"
        line_dir.mkdir()

        line_image = Image.fromarray((~line * 255).astype(np.uint8), mode="L")
        line_image.save(line_dir / "line.png")

        word_regions = word_segmenter.find_regions(line)

        print(f"Number of words: {len(word_regions)}")

        words = word_segmenter.crop(line, word_regions)

        for word_index, word in enumerate(words):
            word_dir = line_dir / f"word_{word_index}"
            word_dir.mkdir()

            word_image = Image.fromarray((~word * 255).astype(np.uint8), mode="L")
            word_image.save(word_dir / "word.png")

            print(f"\nWord {word_index}: shape={word.shape}")

            character_regions = character_segmenter.find_regions(word)

            print(f"Initial character regions: {len(character_regions)}")

            conjoined_candidates = conjoined_segmenter.find_candidates(
                word,
                character_regions,
            )

            print(f"Conjoined candidates: {len(conjoined_candidates)}")

            processed_regions = conjoined_segmenter.process(
                word,
                character_regions,
                model,
                device,
            )

            print(f"Final character regions: {len(processed_regions)}")

            characters = character_segmenter.crop(
                word,
                processed_regions,
            )

            for character_index, character in enumerate(characters):
                character.save(
                    word_dir / f"character_{character_index}.png"
                )

                prediction, confidence = conjoined_segmenter.predict(
                    model,
                    character,
                    device,
                )

                predicted_character = class_names[prediction] if 0 <= prediction < len(class_names) else "?"

                print(
                    f"  Character {character_index}: "
                    f"{predicted_character} "
                    f"(confidence={confidence:.3f})"
                )