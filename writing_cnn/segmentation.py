import numpy as np
import torch
from PIL import Image

from math_cnn.images import crop_and_center_drawing, prepare_symbol_image


class LineSegmenter:
    def __init__(self, min_ink_pixels=8, max_internal_gap=2, min_line_height=10):
        self.min_ink_pixels = min_ink_pixels
        self.max_internal_gap = max_internal_gap
        self.min_line_height = min_line_height

    def find_regions(self, binary):
        projection = np.sum(binary, axis=1)
        active = projection >= self.min_ink_pixels
        regions, start, gap = [], None, 0

        for y, ink in enumerate(active):
            if ink:
                if start is None:
                    start = y
                gap = 0
            elif start is not None:
                gap += 1
                if gap > self.max_internal_gap:
                    end = y - gap
                    if end - start >= self.min_line_height:
                        regions.append((start, end))
                    start, gap = None, 0

        if start is not None:
            end = len(active) - gap
            if end - start >= self.min_line_height:
                regions.append((start, end))

        return regions

    def crop(self, binary, regions):
        return [self.crop_to_ink(binary[y1:y2, :]) for y1, y2 in regions]

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)
        if len(xs) == 0:
            raise ValueError("No handwriting detected.")
        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class WordSegmenter:
    def __init__(self, word_gap_threshold=8):
        self.word_gap_threshold = word_gap_threshold

    def find_regions(self, line):
        projection = np.sum(line, axis=0)
        active = projection > 0
        regions, start, gap_start = [], None, None

        for x, has_ink in enumerate(active):
            if has_ink:
                if start is None:
                    start = x
                gap_start = None
            elif start is not None:
                if gap_start is None:
                    gap_start = x
                if x - gap_start > self.word_gap_threshold:
                    regions.append((start, gap_start))
                    start, gap_start = None, None

        if start is not None:
            regions.append((start, len(active)))

        return regions

    def crop(self, line, regions):
        return [self.crop_to_ink(line[:, x1:x2]) for x1, x2 in regions]

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)
        if len(xs) == 0:
            raise ValueError("No handwriting detected.")
        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class CharacterSegmenter:
    def find_regions(self, word):
        projection = np.sum(word, axis=0)
        active = projection > 0
        regions, start = [], None

        for x, has_ink in enumerate(active):
            if has_ink and start is None:
                start = x
            elif not has_ink and start is not None:
                regions.append((start, x))
                start = None

        if start is not None:
            regions.append((start, len(active)))

        return regions

    def crop(self, word, regions):
        characters = []

        for x1, x2 in regions:
            image = Image.fromarray((~word[:, x1:x2] * 255).astype(np.uint8), mode="L")
            image = crop_and_center_drawing(image)
            if image is not None:
                characters.append(image)

        return characters


class ConjoinedCharacterSegmenter:
    def __init__(self, width_multiplier=1.5, min_character_width=5):
        self.width_multiplier = width_multiplier
        self.min_character_width = min_character_width
        self.normal_width = -1

    def find_candidates(self, word, character_regions):
        if len(character_regions) < 2:
            return []

        widths = np.array([x2 - x1 for x1, x2 in character_regions], dtype=float)
        sorted_widths = np.sort(widths)
        normal_count = max(1, len(sorted_widths) // 2)
        self.normal_width = float(np.median(sorted_widths[:normal_count]))
        threshold = max(self.min_character_width * 2, self.normal_width * self.width_multiplier)

        return [region for region in character_regions if region[1] - region[0] >= threshold]

    def find_split_candidates(self, word, region, num_candidates=5, smoothing_window=3):
        x1, x2 = region
        width = x2 - x1

        if width < self.min_character_width * 2:
            return []

        projection = np.sum(word[:, x1:x2], axis=0).astype(float)

        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            projection = np.convolve(projection, kernel, mode="same")

        normal_width = self.normal_width or self.min_character_width
        minimum_width = max(self.min_character_width, int(normal_width * 0.5))
        candidates = []

        for i in range(minimum_width, width - minimum_width):
            left = projection[i - 1]
            current = projection[i]
            right = projection[i + 1]

            if current > left or current > right:
                continue

            window = max(2, smoothing_window)
            start = max(0, i - window)
            end = min(width, i + window + 1)
            surrounding = (np.mean(projection[start:i]) + np.mean(projection[i + 1:end])) / 2

            if surrounding <= 0:
                continue

            depth = 1.0 - current / surrounding

            if depth < 0.10:
                continue

            candidates.append((depth, x1 + i))

        candidates.sort(reverse=True)

        selected = []
        spacing = max(smoothing_window, int(normal_width * 0.2))

        for _, position in candidates:
            if all(abs(position - existing) >= spacing for existing in selected):
                selected.append(position)

            if len(selected) >= num_candidates:
                break

        return selected

    def evaluate_split(self, word, region, split, model, device):
        x1, x2 = region
        left_width = split - x1
        right_width = x2 - split
        width_ratio = min(left_width, right_width) / max(left_width, right_width)

        if width_ratio < 0.2:
            return 0.0, -1, -1, 0.0, 0.0

        left = self.crop_to_ink(word[:, x1:split])
        right = self.crop_to_ink(word[:, split:x2])

        left_height, right_height = left.shape[0], right.shape[0]
        left_area, right_area = np.count_nonzero(left), np.count_nonzero(right)

        normal_width = self.normal_width or self.min_character_width
        normal_height = word.shape[0]

        min_height = int(normal_height * 0.35)
        min_area = max(4, int(normal_width * normal_height * 0.02))

        if left_height < min_height or right_height < min_height:
            return 0.0, -1, -1, 0.0, 0.0

        if left_area < min_area or right_area < min_area:
            return 0.0, -1, -1, 0.0, 0.0

        left = self.crop_to_ink(word[:, x1:split])
        right = self.crop_to_ink(word[:, split:x2])

        left_image = crop_and_center_drawing(Image.fromarray((~left * 255).astype(np.uint8), mode="L"))
        right_image = crop_and_center_drawing(Image.fromarray((~right * 255).astype(np.uint8), mode="L"))

        if left_image is None or right_image is None:
            return 0.0, -1, -1, 0.0, 0.0

        left_prediction, left_confidence = self.predict(model, left_image, device)
        right_prediction, right_confidence = self.predict(model, right_image, device)

        confidence = np.sqrt(left_confidence * right_confidence)

        start = max(0, split - x1 - 3)
        end = min(x2 - x1, split - x1 + 4)
        projection = np.sum(word[:, x1:x2], axis=0).astype(float)
        local = projection[start:end]
        valley_depth = 1.0 - projection[split - x1] / max(np.mean(local), 1.0)
        valley_score = np.clip(0.5 + 0.5 * valley_depth, 0.5, 1.0)

        score = confidence * valley_score

        return score, left_prediction, right_prediction, left_confidence, right_confidence

    def evaluate_unsplit(self, word, region, model, device):
        x1, x2 = region
        character = self.crop_to_ink(word[:, x1:x2])
        image = crop_and_center_drawing(Image.fromarray((~character * 255).astype(np.uint8), mode="L"))

        if image is None:
            return 0.0, -1

        prediction, confidence = self.predict(model, image, device)
        return confidence, prediction

    def should_split(self, split_score, unsplit_confidence, region_width, required_improvement=0.05, minimum_split_score=0.50):
        normal_width = self.normal_width or self.min_character_width

        width_ratio = region_width / normal_width
        width_penalty = min(1.0, (1.5 / width_ratio) ** 0.5)

        adjusted_unsplit = unsplit_confidence * width_penalty

        return (
            split_score >= minimum_split_score and
            split_score - adjusted_unsplit >= required_improvement
        )

    def find_best_split(self, word, region, model, device):
        unsplit_confidence, _ = self.evaluate_unsplit(word, region, model, device)
        best_split = None
        best_score = 0.0
        best_left = -1
        best_right = -1

        for split in self.find_split_candidates(word, region):
            score, left, right, left_conf, right_conf = self.evaluate_split(word, region, split, model, device)

            print(f" Split {split}: score={score:.3f}, left={left} ({left_conf:.3f}), right={right} ({right_conf:.3f})")

            if score > best_score:
                best_split = split
                best_score = score
                best_left = left
                best_right = right

        print(f" Best split: {best_split}, score={best_score:.3f}, unsplit={unsplit_confidence:.3f}")

        should_split = self.should_split(best_score, unsplit_confidence, region[1] - region[0])
        normal_width = self.normal_width or self.min_character_width
        width_ratio = (region[1] - region[0]) / normal_width
        width_penalty = min(1.0, (1.5 / width_ratio) ** 0.5)
        adjusted_unsplit = unsplit_confidence * width_penalty

        print(f" Best split: {best_split}, score={best_score:.3f}, unsplit={unsplit_confidence:.3f}, adjusted_unsplit={adjusted_unsplit:.3f}")
        print(f" Should split: {should_split}")
        if best_split is None or not should_split:
            return None, best_score, best_left, best_right

        print(f" Applying split at {best_split}")
        return best_split, best_score, best_left, best_right

    def process(self, word, character_regions, model, device):
        regions = character_regions.copy()

        for region in self.find_candidates(word, regions):
            split, _, _, _ = self.find_best_split(word, region, model, device)

            if split is None:
                continue

            index = regions.index(region)
            regions[index:index + 1] = [(region[0], split), (split, region[1])]

        return regions

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)

        if len(xs) == 0:
            raise ValueError("No handwriting detected.")

        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    @staticmethod
    def predict(model, character, device):
        tensor = prepare_symbol_image(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()

class SegmentationPipeline:
    def __init__(self, line_segmenter, word_segmenter, character_segmenter, conjoined_segmenter=None):
        self.line_segmenter = line_segmenter
        self.word_segmenter = word_segmenter
        self.character_segmenter = character_segmenter
        self.conjoined_segmenter = conjoined_segmenter

    @staticmethod
    def binarize(image, threshold=200):
        return np.array(image.convert("L")) < threshold

    @staticmethod
    def find_writing_region(binary):
        ys, xs = np.where(binary)
        if len(xs) == 0:
            raise ValueError("No handwriting detected.")
        return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def segment(self, image, model=None, device=None):
        binary = self.binarize(image)
        writing = self.find_writing_region(binary)
        lines = self.line_segmenter.crop(writing, self.line_segmenter.find_regions(writing))
        result = []

        for line in lines:
            words = self.word_segmenter.crop(line, self.word_segmenter.find_regions(line))
            line_result = []

            for word in words:
                regions = self.character_segmenter.find_regions(word)

                if self.conjoined_segmenter is not None:
                    if model is None or device is None:
                        raise ValueError("Model and device are required for conjoined-character processing.")
                    regions = self.conjoined_segmenter.process(word, regions, model, device)

                line_result.append(self.character_segmenter.crop(word, regions))

            result.append(line_result)

        return result