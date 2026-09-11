import cv2
import numpy as np
import torch
from PIL import Image

from math_cnn.images import crop_and_center_drawing, prepare_symbol_image
from writing_cnn.prediction import PredictionPipeline


# Line segmentation

class LineSegmenter:
    def __init__(self, min_ink_pixels=8, max_internal_gap=2, min_line_height=10):
        self.min_ink_pixels = min_ink_pixels
        self.max_internal_gap = max_internal_gap
        self.min_line_height = min_line_height

    def find_regions(self, binary):
        projection = np.sum(binary, axis=1)
        nonzero = projection[projection > 0]

        if not len(nonzero):
            return []

        image_height = binary.shape[0]
        minimum_ink = min(self.min_ink_pixels, max(2, round(float(np.percentile(nonzero, 30)))))
        maximum_gap = max(self.max_internal_gap, round(image_height * 0.005))
        minimum_height = min(self.min_line_height, max(2, round(image_height * 0.05)))
        core_regions = self._find_active_regions(projection >= minimum_ink, maximum_gap, minimum_height)

        if not core_regions:
            return self._find_active_regions(projection > 0, maximum_gap, 1)

        # Dense rows locate distinct line bodies. Each body then owns half of
        # the space to its neighbours, allowing sparse ascenders, descenders,
        # and dots to be restored without joining two lines.
        regions = []
        for index, (start, end) in enumerate(core_regions):
            top = 0 if index == 0 else (core_regions[index - 1][1] + start + 1) // 2
            bottom = image_height if index + 1 == len(core_regions) else (end + core_regions[index + 1][0] + 1) // 2
            ink_rows = np.flatnonzero(projection[top:bottom] > 0)

            if len(ink_rows):
                regions.append((top + int(ink_rows[0]), top + int(ink_rows[-1]) + 1))

        return regions

    @staticmethod
    def _find_active_regions(active, maximum_gap, minimum_height):
        regions, start, gap = [], None, 0

        for y, ink in enumerate(active):
            if ink:
                if start is None:
                    start = y
                gap = 0
            elif start is not None:
                gap += 1
                if gap > maximum_gap:
                    end = y - gap + 1
                    if end - start >= minimum_height:
                        regions.append((start, end))
                    start, gap = None, 0

        if start is not None:
            end = len(active) - gap
            if end - start >= minimum_height:
                regions.append((start, end))

        return regions

    def crop(self, binary, regions):
        return [self.crop_to_ink(binary[y1:y2]) for y1, y2 in regions]

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)
        if xs.size == 0:
            raise ValueError("No handwriting detected.")
        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# Word segmentation

class WordSegmenter:
    def __init__(
        self,
        word_gap_threshold=8,
        gap_multiplier=1.7,
        min_gap_threshold=2,
        min_gap_ratio=1.45,
        strong_gap_ratio=2.0,
        min_word_width=8,
    ):
        self.word_gap_threshold = word_gap_threshold
        self.gap_multiplier = gap_multiplier
        self.min_gap_threshold = min_gap_threshold
        self.min_gap_ratio = min_gap_ratio
        self.strong_gap_ratio = strong_gap_ratio
        self.min_word_width = min_word_width

    @staticmethod
    def find_gaps(line):
        active = np.any(line, axis=0)
        gaps, start = [], None

        for x, ink in enumerate(active):
            if ink:
                if start is not None:
                    gaps.append((start, x - start))
                start = None
            elif start is None:
                start = x

        return [(x, length) for x, length in gaps if length > 0]

    @staticmethod
    def find_runs(line):
        active = np.any(line, axis=0)
        runs, start = [], None

        for x, ink in enumerate(active):
            if ink and start is None:
                start = x
            elif not ink and start is not None:
                runs.append((start, x))
                start = None

        if start is not None:
            runs.append((start, len(active)))

        return runs

    @staticmethod
    def gap_statistics(gaps):
        values = np.asarray(
            [length for _, length in gaps],
            dtype=float,
        )

        if not len(values):
            return 0.0, 0.0

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad

    @staticmethod
    def normalized_gaps(gaps, runs, line_height):
        if not gaps:
            return []

        widths = np.asarray(
            [x2 - x1 for x1, x2 in runs],
            dtype=float,
        )

        width = float(np.median(widths)) if len(widths) else 1.0
        scale = max(1.0, min(line_height * 0.35, width))

        result = []

        for index, (x, gap) in enumerate(gaps):
            left_width = runs[index][1] - runs[index][0] if index < len(runs) else width
            right_width = runs[index + 1][1] - runs[index + 1][0] if index + 1 < len(runs) else width
            local_width = max(1.0, min(left_width, right_width, width))
            local_scale = max(scale, local_width * 0.25)

            result.append(
                (
                    x,
                    gap,
                    gap / local_scale,
                    gap / max(float(line_height), 1.0),
                )
            )

        return result

    def largest_gap_split(self, gaps, runs, line_height):
        if len(gaps) < 2:
            return []

        normalized = self.normalized_gaps(
            gaps,
            runs,
            line_height,
        )

        ordered = sorted(
            normalized,
            key=lambda value: value[1],
            reverse=True,
        )

        median = float(
            np.median(
                [item[1] for item in normalized]
            )
        )

        candidates = []

        for x, gap, ratio, height_ratio in ordered:
            if gap < self.word_gap_threshold:
                continue

            if gap < median * self.min_gap_ratio:
                continue

            candidates.append(
                (x, gap, ratio, height_ratio)
            )

        if not candidates:
            return []

        largest = candidates[0]

        if largest[2] < self.min_gap_ratio and largest[3] < 0.15:
            return []

        boundaries = [largest]

        for candidate in candidates[1:]:
            if candidate[0] == largest[0]:
                continue

            if candidate[1] >= largest[1] * 0.75 and candidate[2] >= self.min_gap_ratio:
                boundaries.append(candidate)

        return sorted(
            boundaries,
            key=lambda value: value[0],
        )

    def find_threshold(self, gaps, runs, line_height):
        if not gaps:
            return float('inf')

        lengths = np.asarray([length for _, length in gaps], dtype=float)
        scale_floor = max(float(self.min_gap_threshold), line_height * 0.12)
        if len(lengths) < 3:
            return scale_floor

        # Cluster in log space so the decision depends on relative spacing rather
        # than the image resolution or one unusually large gap.
        values = np.log1p(lengths)
        centers = np.quantile(values, (0.25, 0.75))
        assignments = np.zeros(len(values), dtype=int)
        for _ in range(20):
            assignments = np.abs(values[:, None] - centers).argmin(axis=1)
            updated = np.asarray([
                values[assignments == cluster].mean()
                if np.any(assignments == cluster)
                else centers[cluster]
                for cluster in range(2)
            ])
            if np.allclose(updated, centers):
                break
            centers = updated

        order = np.argsort(centers)
        small = lengths[assignments == order[0]]
        large = lengths[assignments == order[1]]
        if not len(small) or not len(large):
            return scale_floor

        separation = np.median(large) / max(np.median(small), 1.0)
        if separation < self.min_gap_ratio:
            return scale_floor

        cluster_boundary = (float(np.max(small)) + float(np.min(large))) * 0.5
        return max(scale_floor, cluster_boundary)

    def find_boundaries(self, line):
        runs = self.find_runs(line)

        if len(runs) < 2:
            return []

        gaps = self.find_gaps(line)

        if not gaps:
            return []

        line_height = line.shape[0]
        threshold = self.find_threshold(
            gaps,
            runs,
            line_height,
        )

        normalized = self.normalized_gaps(gaps, runs, line_height)
        median_gap = max(float(np.median([item[1] for item in normalized])), 1.0)
        boundaries = [
            (x, gap, gap / median_gap, local_ratio, height_ratio)
            for x, gap, local_ratio, height_ratio in normalized
            if gap >= threshold
        ]

        if not boundaries:
            boundaries = self.largest_gap_split(
                gaps,
                runs,
                line_height,
            )

        if not boundaries:
            return []

        boundaries.sort(key=lambda value: value[0])

        filtered = []

        for boundary in boundaries:
            if not filtered:
                filtered.append(boundary)
                continue

            spacing = max(
                2,
                int(line_height * 0.05),
            )

            if boundary[0] - filtered[-1][0] < spacing:
                if boundary[2] > filtered[-1][2]:
                    filtered[-1] = boundary
            else:
                filtered.append(boundary)

        return filtered

    def find_regions(self, line):
        if line.size == 0 or not np.any(line):
            return []

        active = np.any(line, axis=0)
        ink = np.flatnonzero(active)

        if not len(ink):
            return []

        start = int(ink[0])
        end = int(ink[-1] + 1)
        boundaries = self.find_boundaries(line)

        if not boundaries:
            return [(start, end)]

        regions = []

        for boundary in boundaries:
            x = int(boundary[0])

            if x <= start or x >= end:
                continue

            if x - start < self.min_word_width:
                continue

            regions.append((start, x))
            start = x

            remaining = ink[ink >= start]

            if len(remaining):
                start = int(remaining[0])

        if start < end and end - start >= self.min_word_width:
            regions.append((start, end))

        return [
            (x1, x2)
            for x1, x2 in regions
            if x2 - x1 >= self.min_word_width
        ]

    def crop(self, line, regions):
        return [
            line[:, x1:x2]
            for x1, x2 in regions
            if x1 < x2 and np.any(line[:, x1:x2])
        ]

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)

        if xs.size == 0:
            raise ValueError("No handwriting detected.")

        return image[
            ys.min():ys.max() + 1,
            xs.min():xs.max() + 1,
        ]


# Character segmentation

class CharacterSegmenter:
    def find_regions(self, word):
        projection = np.sum(word, axis=0)
        active = projection > 0
        regions, start = [], None

        for x, ink in enumerate(active):
            if ink and start is None:
                start = x
            elif not ink and start is not None:
                regions.append((start, x))
                start = None

        if start is not None:
            regions.append((start, len(active)))

        return regions

    def crop(self, word, regions):
        characters = []

        for x1, x2 in regions:
            character = word[:, x1:x2]
            ys, xs = np.where(character)
            if xs.size == 0:
                continue

            geometry = {
                "height_ratio": (int(ys.max()) - int(ys.min()) + 1) / word.shape[0],
                "top_ratio": int(ys.min()) / word.shape[0],
                "bottom_ratio": (int(ys.max()) + 1) / word.shape[0],
                "width_to_height": (int(xs.max()) - int(xs.min()) + 1) / word.shape[0],
                "components": cv2.connectedComponents(character.astype(np.uint8), connectivity=8)[0] - 1,
            }
            image = Image.fromarray((~character * 255).astype(np.uint8), mode="L")

            image = crop_and_center_drawing(image)

            if image is not None:
                image.info["geometry"] = geometry
                characters.append(image)

        return characters


# Conjoined character segmentation

class ConjoinedCharacterSegmenter:
    def __init__(self, width_multiplier=1.5, min_character_width=5, geometry_split_ratio=2.2):
        self.width_multiplier = width_multiplier
        self.min_character_width = min_character_width
        self.geometry_split_ratio = geometry_split_ratio
        self.normal_width = -1

    def find_candidates(self, word, character_regions):
        if not character_regions:
            return []

        widths = np.asarray([x2 - x1 for x1, x2 in character_regions], dtype=float)
        height_prior = max(float(self.min_character_width), word.shape[0] * 0.25)
        plausible_widths = widths[widths >= height_prior * 0.35]

        if len(plausible_widths) >= 2:
            lower_half = np.sort(plausible_widths)[:max(1, len(plausible_widths) // 2)]
            observed_width = float(np.median(lower_half))
            self.normal_width = float(np.clip(observed_width, height_prior * 0.65, height_prior * 1.2))
        else:
            # Fully cursive words often arrive as one connected component. In
            # that case line height provides a resolution-independent width prior.
            self.normal_width = height_prior

        threshold = max(
            self.min_character_width * 2,
            self.normal_width * self.width_multiplier,
        )

        return [
            region
            for region in character_regions
            if region[1] - region[0] >= threshold
        ]

    def find_split_candidates(self, word, region, num_candidates=10, smoothing_window=3):
        x1, x2 = region
        width = x2 - x1

        if width < self.min_character_width * 2:
            return []

        projection = np.sum(
            word[:, x1:x2],
            axis=0,
        ).astype(float)

        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            projection = np.convolve(
                projection,
                kernel,
                mode="same",
            )

        normal_width = self.normal_width or self.min_character_width
        minimum_width = max(
            self.min_character_width,
            int(normal_width * 0.5),
        )

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

            surrounding = (
                np.mean(projection[start:i])
                + np.mean(projection[i + 1:end])
            ) / 2

            if surrounding <= 0:
                continue

            depth = 1.0 - current / surrounding

            if depth >= 0.10:
                candidates.append(
                    (depth, x1 + i)
                )

        candidates.sort(reverse=True)

        selected = []
        spacing = max(
            smoothing_window,
            int(normal_width * 0.2),
        )

        for _, position in candidates:
            if all(
                abs(position - existing) >= spacing
                for existing in selected
            ):
                selected.append(position)

            if len(selected) >= num_candidates:
                break

        return selected

    @staticmethod
    def crop_to_ink(image):
        ys, xs = np.where(image)

        if xs.size == 0:
            raise ValueError("No handwriting detected.")

        return image[
            ys.min():ys.max() + 1,
            xs.min():xs.max() + 1,
        ]

    @staticmethod
    def predict(model, image, device):
        image = PredictionPipeline.normalize_stroke_thickness(image)
        tensor = prepare_symbol_image(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        confidence, prediction = probabilities.max(dim=1)

        return prediction.item(), confidence.item()

    @staticmethod
    def predict_topk(model, image, device, k=3):
        image = PredictionPipeline.normalize_stroke_thickness(image)
        tensor = prepare_symbol_image(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        k = min(k, probabilities.shape[1])
        confidence, prediction = torch.topk(probabilities, k, dim=1)

        return [
            (p.item(), c.item())
            for p, c in zip(prediction[0], confidence[0])
        ]

    def evaluate_split(self, word, region, split, model, device):
        x1, x2 = region
        left_width = split - x1
        right_width = x2 - split
        normal_width = self.normal_width or self.min_character_width

        if min(left_width, right_width) < self.min_character_width:
            return 0.0, -1, -1, 0.0, 0.0

        width_score = min(
            left_width / normal_width,
            right_width / normal_width,
            1.0,
        )

        if width_score < 0.35:
            return 0.0, -1, -1, 0.0, 0.0

        left = self.crop_to_ink(word[:, x1:split])
        right = self.crop_to_ink(word[:, split:x2])

        minimum_area = max(
            4,
            int(normal_width * word.shape[0] * 0.02),
        )

        if min(
            np.count_nonzero(left),
            np.count_nonzero(right),
        ) < minimum_area:
            return 0.0, -1, -1, 0.0, 0.0

        left = crop_and_center_drawing(
            Image.fromarray(
                (~left * 255).astype(np.uint8),
                mode="L",
            )
        )

        right = crop_and_center_drawing(
            Image.fromarray(
                (~right * 255).astype(np.uint8),
                mode="L",
            )
        )

        if left is None or right is None:
            return 0.0, -1, -1, 0.0, 0.0

        lc = self.predict_topk(model, left, device, 3)
        rc = self.predict_topk(model, right, device, 3)

        if not lc or not rc:
            return 0.0, -1, -1, 0.0, 0.0

        lp, lconf = lc[0]
        rp, rconf = rc[0]

        cnn_score = np.sqrt(
            lconf * rconf
        )

        projection = np.sum(
            word[:, x1:x2],
            axis=0,
        )

        position = split - x1
        window = max(
            2,
            int(normal_width * 0.15),
        )

        around = np.r_[
            projection[
                max(0, position - window):position
            ],
            projection[
                position + 1:min(
                    len(projection),
                    position + window + 1,
                )
            ],
        ]

        valley = np.clip(
            1.0
            - projection[position]
            / max(
                np.median(around),
                1.0,
            ),
            0.0,
            1.0,
        )

        score = (
            cnn_score
            * (0.5 + 0.5 * valley)
            * np.sqrt(width_score)
        )

        return (
            score,
            lp,
            rp,
            lconf,
            rconf,
        )

    def evaluate_unsplit(self, word, region, model, device):
        x1, x2 = region

        character = self.crop_to_ink(
            word[:, x1:x2]
        )

        image = crop_and_center_drawing(
            Image.fromarray(
                (~character * 255).astype(np.uint8),
                mode="L",
            )
        )

        if image is None:
            return 0.0, -1

        prediction, confidence = self.predict(
            model,
            image,
            device,
        )

        return confidence, prediction

    def should_split(
        self,
        split_score,
        unsplit_confidence,
        region_width,
        force_geometry=False,
        required_improvement=0.05,
        minimum_split_score=0.30,
    ):
        normal_width = self.normal_width or self.min_character_width
        width_ratio = region_width / normal_width
        width_penalty = min(1.0, (1.5 / width_ratio) ** 0.5)
        adjusted_unsplit = unsplit_confidence * width_penalty

        # A classifier can be confidently wrong when several joined letters
        # resemble one EMNIST character. Strong geometric evidence prevents that
        # confidence from blocking every split in fully cursive words.
        geometry_requires_split = (
            force_geometry
            and width_ratio >= self.geometry_split_ratio
            and split_score >= minimum_split_score * 0.8
        )

        return (
            geometry_requires_split
            or (
                split_score >= minimum_split_score
                and split_score - adjusted_unsplit >= required_improvement
            )
        )

    def find_best_split(self, word, region, model, device, force_geometry=False):
        unsplit_confidence, _ = self.evaluate_unsplit(
            word,
            region,
            model,
            device,
        )

        best_split = None
        best_score = 0.0
        best_left = -1
        best_right = -1

        for split in self.find_split_candidates(
            word,
            region,
        ):
            score, left, right, left_conf, right_conf = self.evaluate_split(
                word,
                region,
                split,
                model,
                device,
            )

            if score > best_score:
                best_split = split
                best_score = score
                best_left = left
                best_right = right

        should_split = (
            best_split is not None
            and self.should_split(
                best_score,
                unsplit_confidence,
                region[1] - region[0],
                force_geometry,
            )
        )

        if not should_split:
            return None, best_score, best_left, best_right

        return (best_split, best_score, best_left, best_right)

    def process(self, word, character_regions, model, device):
        regions = character_regions.copy()

        while True:
            candidates = self.find_candidates(word, regions)

            if not candidates:
                break
            changed = False

            for region in candidates:
                split, _, _, _ = self.find_best_split(word, region, model, device, force_geometry=True)
                if split is None:
                    continue

                index = regions.index(region)
                regions[index:index + 1] = [
                    (region[0], split),
                    (split, region[1]),
                ]

                changed = True
                break
            
            if not changed:
                break
        return regions


# Segmentation pipeline

class SegmentationPipeline:
    def __init__(self, line_segmenter, word_segmenter, character_segmenter, conjoined_segmenter=None):
        self.line_segmenter = line_segmenter
        self.word_segmenter = word_segmenter
        self.character_segmenter = character_segmenter
        self.conjoined_segmenter = conjoined_segmenter

    @staticmethod
    def normalize_resolution(binary, target_height=24):
        """Give segmentation a consistent ink scale, independent of page margins."""
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        components = stats[1:]
        if not len(components):
            return binary
        # Ignore tiny specks and dots when estimating the main writing size.
        substantial = components[components[:, cv2.CC_STAT_AREA] >= components[:, cv2.CC_STAT_AREA].max() * 0.01]
        height = float(np.percentile(substantial[:, cv2.CC_STAT_HEIGHT], 75))
        scale = target_height / max(height, 1.0)
        output_size = (max(1, round(binary.shape[1] * scale)), max(1, round(binary.shape[0] * scale)))
        resized = cv2.resize(binary.astype(np.float32), output_size, interpolation=cv2.INTER_AREA)
        return resized >= 0.5

    @staticmethod
    def binarize(image, threshold=None):
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        if threshold is None:
            _, binary = cv2.threshold(grayscale, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = (grayscale < threshold).astype(np.uint8)
        binary = SegmentationPipeline.normalize_resolution(binary.astype(bool))
        return SegmentationPipeline.remove_noise(binary)

    @staticmethod
    def remove_noise(binary):
        height, width = binary.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(binary, dtype=bool)
        minimum_area = max(2, round(min(height, width) * 0.005))
        vertical_edge_width = max(2, round(width * 0.01))
        horizontal_edge_height = max(2, round(height * 0.01))

        for label in range(1, count):
            x, y, component_width, component_height, area = map(int, stats[label])
            touches_vertical_edge = x == 0 or x + component_width == width
            touches_horizontal_edge = y == 0 or y + component_height == height
            thin_vertical_edge_mark = touches_vertical_edge and component_width <= vertical_edge_width
            thin_horizontal_edge_mark = touches_horizontal_edge and component_height <= horizontal_edge_height
            if area < minimum_area or thin_vertical_edge_mark or thin_horizontal_edge_mark:
                continue
            cleaned[labels == label] = True

        return cleaned

    @staticmethod
    def find_writing_region(binary):
        ys, xs = np.where(binary)

        if xs.size == 0:
            raise ValueError("No handwriting detected.")

        return binary[
            ys.min():ys.max() + 1,
            xs.min():xs.max() + 1,
        ]

    def find_words(self, image):
        binary = self.binarize(image)
        writing = self.find_writing_region(binary)
        lines = self.line_segmenter.crop(
            writing,
            self.line_segmenter.find_regions(writing),
        )
        return [self.word_segmenter.crop(line, self.word_segmenter.find_regions(line)) for line in lines]

    def segment_binary_word(self, word, conjoined_segmenter=None, model=None, device=None):
        regions = self.character_segmenter.find_regions(word)
        if conjoined_segmenter is not None:
            if model is None or device is None:
                raise ValueError('Model and device are required for conjoined-character processing.')
            regions = conjoined_segmenter.process(word, regions, model, device)
        return self.character_segmenter.crop(word, regions)

    def segment(self, image, model=None, device=None):
        return [
            [self.segment_binary_word(word, self.conjoined_segmenter, model, device) for word in line]
            for line in self.find_words(image)
        ]

    def segment_hypotheses(self, image, conjoined_segmenters, model=None, device=None):
        words = self.find_words(image)
        return [
            [[self.segment_binary_word(word, segmenter, model, device) for word in line] for line in words]
            for segmenter in conjoined_segmenters
        ]

    def segment_word(self, image, model=None, device=None):
        """Segment one already-cropped word without rediscovering page layout."""
        word = self.find_writing_region(self.binarize(image))
        return self.segment_binary_word(word, self.conjoined_segmenter, model, device)

    def segment_word_hypotheses(self, image, conjoined_segmenters, model=None, device=None):
        word = self.find_writing_region(self.binarize(image))
        return [self.segment_binary_word(word, segmenter, model, device) for segmenter in conjoined_segmenters]
