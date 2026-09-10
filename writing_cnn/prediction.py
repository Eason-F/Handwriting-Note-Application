from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from wordfreq import top_n_list, zipf_frequency

from math_cnn.images import prepare_symbol_image


# Language model

class CharacterLanguageModel:
    def __init__(self, vocabulary_size=50000, n=4, smoothing=0.1):
        self.n = n
        self.smoothing = smoothing
        self.counts = Counter()
        self.contexts = Counter()
        self._build(vocabulary_size)

    def _build(self, vocabulary_size):
        for word in top_n_list("en", vocabulary_size, wordlist="best"):
            word = "".join(c for c in word.lower() if c.isalpha())

            if not word:
                continue

            frequency = zipf_frequency(word, "en", wordlist="best")
            weight = max(1, int(10 ** max(frequency - 3, 0)))
            text = "^" * (self.n - 1) + word + "$"

            for i in range(len(text) - self.n + 1):
                context = text[i:i + self.n - 1]
                character = text[i + self.n - 1]
                self.counts[(context, character)] += weight
                self.contexts[context] += weight

    def next_log_probability(self, prefix, character):
        context = ("^" * (self.n - 1) + prefix)[-(self.n - 1):]
        numerator = self.counts[(context, character)] + self.smoothing
        denominator = self.contexts[context] + self.smoothing * 27
        return float(np.log(numerator / denominator))

    def score(self, word):
        word = "".join(c for c in word.lower() if c.isalpha())

        if not word:
            return -20.0

        score = 0.0
        prefix = ""

        for character in word:
            score += self.next_log_probability(prefix, character)
            prefix += character

        score += self.next_log_probability(prefix, "$")
        return float(score)


# Prediction pipeline

class PredictionPipeline:
    VISUAL_ALTERNATIVES = {
        "0": "o",
        "1": "li",
        "2": "z",
        "4": "a",
        "5": "s",
        "6": "gb",
        "7": "t",
        "8": "b",
        "9": "gq",
    }

    def __init__(self, characters, lm_vocabulary_size=50000):
        self.characters = characters
        self._vocabulary = None
        self._vocabulary_by_length = {}
        self.language_model = CharacterLanguageModel(lm_vocabulary_size)

    # Preprocessing

    @staticmethod
    def estimate_stroke_width(ink):
        if not np.any(ink):
            return 0.0

        distance = cv2.distanceTransform(
            ink.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        values = distance[distance > 0]

        return float(np.median(values) * 2) if len(values) else 0.0

    @staticmethod
    def normalize_stroke_thickness(image, target_width=3.0, tolerance=0.5):
        metadata = image.info.copy()
        image_array = np.asarray(image.convert("L"))
        ink = image_array < 128

        if not np.any(ink):
            return image

        kernel = np.ones((3, 3), np.uint8)

        for _ in range(3):
            width = PredictionPipeline.estimate_stroke_width(ink)

            if abs(width - target_width) <= tolerance:
                break

            if width < target_width:
                ink = cv2.dilate(
                    ink.astype(np.uint8),
                    kernel,
                    1,
                ).astype(bool)
                continue

            eroded = cv2.erode(
                ink.astype(np.uint8),
                kernel,
                1,
            )

            if not np.any(eroded):
                break

            ink = eroded.astype(bool)

        output = np.full_like(image_array, 255)
        output[ink] = 0

        normalized = Image.fromarray(output)
        normalized.info.update(metadata)
        return normalized

    @staticmethod
    def prepare_character(character):
        character = PredictionPipeline.normalize_stroke_thickness(character)
        return prepare_symbol_image(character)

    # Character prediction

    @staticmethod
    def predict(model, character, device):
        tensor = PredictionPipeline.prepare_character(character)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(tensor),
                dim=1,
            )

        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()

    @staticmethod
    def predict_topk(model, character, device, k=10):
        tensor = PredictionPipeline.prepare_character(character)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(tensor),
                dim=1,
            )

        k = min(k, probabilities.shape[1])
        confidence, prediction = torch.topk(
            probabilities,
            k,
            dim=1,
        )

        return [
            (p.item(), c.item())
            for p, c in zip(
                prediction[0],
                confidence[0],
            )
        ]

    def predict_character(self, model, character, device):
        prediction, confidence = self.predict(
            model,
            character,
            device,
        )

        return self.characters[prediction], confidence

    def predict_character_candidates(self, model, character, device, k=10):
        return [
            (self.characters[p], c)
            for p, c in self.predict_topk(
                model,
                character,
                device,
                k,
            )
        ]

    def _character_probabilities(self, model, word, device, k):
        tensors = [
            self.prepare_character(character)
            for character in word
        ]

        if not tensors:
            return []

        batch = torch.stack(tensors).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(batch),
                dim=1,
            )

        k = min(k, probabilities.shape[1])
        values, indices = torch.topk(
            probabilities,
            k,
            dim=1,
        )

        result = []

        for character_image, row_values, row_indices in zip(word, values, indices):
            candidates = {}
            for index, value in zip(row_indices, row_values):
                character = self.characters[index.item()]
                probability = float(value.item())
                candidates[character] = probability

                # Dictionary and language-model candidates are lowercase. Merge
                # case variants instead of silently assigning them near-zero odds.
                lowercase = character.lower()
                if lowercase != character:
                    candidates[lowercase] = candidates.get(lowercase, 0.0) + probability
                else:
                    candidates[lowercase] = max(candidates.get(lowercase, 0.0), probability)

                # EMNIST frequently confuses handwritten letters with visually
                # identical digits. Retain the CNN signal with a modest penalty.
                for alternative in self.VISUAL_ALTERNATIVES.get(character, ""):
                    candidates[alternative] = max(candidates.get(alternative, 0.0), probability * 0.45)
            self.apply_geometry_to_ambiguous_classes(candidates, character_image.info.get("geometry", {}))
            result.append(candidates)

        return result

    @staticmethod
    def apply_geometry_to_ambiguous_classes(candidates, geometry):
        group = ("I", "i", "l", "1")
        evidence = sum(candidates.get(character, 0.0) for character in group)
        if evidence <= 0:
            return

        has_dot = geometry.get("components", 1) >= 2
        if has_dot:
            candidates["i"] = max(candidates.get("i", 0.0), evidence * 0.85)
            candidates["l"] = min(candidates.get("l", 0.0), evidence * 0.20)
        else:
            candidates["l"] = max(candidates.get("l", 0.0), evidence * 0.75)
            candidates["i"] = min(candidates.get("i", 0.0), evidence * 0.30)

    # CNN decoding

    def predict_word_cnn(self, model, word, device, character_top_k=10, beam_width=100):
        probabilities = self._character_probabilities(
            model,
            word,
            device,
            character_top_k,
        )

        beams = [("", 0.0)]

        for candidates in probabilities:
            next_beams = []

            for text, score in beams:
                for character, probability in candidates.items():
                    next_beams.append(
                        (
                            text + character,
                            score + np.log(
                                max(probability, 1e-8)
                            ),
                        )
                    )

            next_beams.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            beams = next_beams[:beam_width]

        return beams

    # WordFreq

    @staticmethod
    def normalize_word(word):
        return "".join(
            c for c in word.lower()
            if c.isalpha()
        )

    @staticmethod
    def wordfreq_score(word):
        word = PredictionPipeline.normalize_word(word)

        if not word:
            return 0.0

        return float(
            zipf_frequency(
                word,
                "en",
                wordlist="best",
            )
        )

    def _load_vocabulary(self, vocabulary_size=50000):
        if self._vocabulary is not None:
            return

        self._vocabulary = {
            word.lower()
            for word in top_n_list(
                "en",
                vocabulary_size,
                wordlist="best",
            )
            if word.isalpha()
        }

        self._vocabulary_by_length = {}

        for word in self._vocabulary:
            self._vocabulary_by_length.setdefault(
                len(word),
                [],
            ).append(word)

    def score_word_against_cnn(self, word, character_candidates):
        score = 0.0

        for character, candidates in zip(
            word,
            character_candidates,
        ):
            probability = candidates.get(
                character,
                1e-8,
            )
            score += np.log(
                max(probability, 1e-8)
            )

        return float(score)

    # Language model

    def lm_score(self, word):
        word = self.normalize_word(word)
        return (
            self.language_model.score(word)
            if word
            else -20.0
        )

    # Decoder

    def predict_word_lm(self, model, word, device, character_top_k=10, beam_width=100, lm_weight=0.15, limit=20):
        probabilities = self._character_probabilities(
            model,
            word,
            device,
            character_top_k,
        )

        beams = [("", 0.0)]

        for candidates in probabilities:
            next_beams = []

            for text, score in beams:
                for character, probability in candidates.items():
                    cnn_score = np.log(
                        max(probability, 1e-8)
                    )

                    lm_score = self.language_model.next_log_probability(
                        text,
                        character,
                    )

                    next_beams.append(
                        (
                            text + character,
                            score
                            + cnn_score
                            + lm_weight * lm_score,
                        )
                    )

            next_beams.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            beams = next_beams[:beam_width]

        return beams[:limit]

    def predict_word_wordfreq(self, model, word, device, character_top_k=10, vocabulary_size=50000, frequency_weight=0.1, limit=20):
        self._load_vocabulary(vocabulary_size)

        probabilities = self._character_probabilities(
            model,
            word,
            device,
            character_top_k,
        )

        candidates = []

        for vocabulary_word in self._vocabulary_by_length.get(
            len(word),
            [],
        ):
            cnn_score = self.score_word_against_cnn(
                vocabulary_word,
                probabilities,
            )

            frequency = self.wordfreq_score(
                vocabulary_word
            )

            candidates.append(
                (
                    vocabulary_word,
                    cnn_score
                    + frequency * frequency_weight,
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return candidates[:limit]

    def predict_word_hybrid(self, model, word, device, character_top_k=10, beam_width=100, frequency_weight=0.1, lm_weight=0.15, limit=20):
        candidates = self.predict_word_lm(
            model,
            word,
            device,
            character_top_k,
            beam_width,
            lm_weight,
            beam_width,
        )

        return self.rerank_word_candidates(
            candidates,
            frequency_weight=frequency_weight,
            lm_weight=0.0,
        )[:limit]

    def predict_word_wordfreq_hybrid(self, model, word, device, character_top_k=10, vocabulary_size=50000, frequency_weight=0.1, lm_weight=0.15, limit=20):
        self._load_vocabulary(vocabulary_size)

        probabilities = self._character_probabilities(
            model,
            word,
            device,
            character_top_k,
        )

        candidates = []

        for vocabulary_word in self._vocabulary_by_length.get(
            len(word),
            [],
        ):
            cnn_score = self.score_word_against_cnn(
                vocabulary_word,
                probabilities,
            )

            lm_score = self.lm_score(
                vocabulary_word
            )

            frequency = self.wordfreq_score(
                vocabulary_word
            )

            score = (
                cnn_score
                + lm_weight * lm_score
                + frequency_weight * frequency
            )

            candidates.append(
                (
                    vocabulary_word,
                    float(score),
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return candidates[:limit]

    # Scoring

    def combined_score(self, word, cnn_score=0.0, frequency_weight=0.1, lm_weight=0.15):
        word = self.normalize_word(word)

        return float(
            cnn_score
            + frequency_weight * self.wordfreq_score(word)
            + lm_weight * self.lm_score(word)
        )

    def rerank_word_candidates(self, candidates, frequency_weight=0.1, lm_weight=0.15):
        reranked = [
            (
                word,
                self.combined_score(
                    word,
                    score,
                    frequency_weight,
                    lm_weight,
                ),
            )
            for word, score in candidates
        ]

        reranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked

    @staticmethod
    def restore_size_based_case(candidates, characters):
        if len(characters) < 2:
            return candidates

        geometries = [character.info.get("geometry", {}) for character in characters]
        heights = [geometry.get("height_ratio", 1.0) for geometry in geometries]
        tops = [geometry.get("top_ratio", 0.0) for geometry in geometries]
        typical_height = float(np.median(heights[1:]))
        typical_top = float(np.median(tops[1:]))
        first_is_capital_sized = heights[0] >= typical_height * 1.15 and tops[0] <= typical_top

        if not first_is_capital_sized:
            return candidates
        return [(word[:1].upper() + word[1:], score) for word, score in candidates]

    # Public word prediction

    def predict_word(self, model, word, device, method="hybrid", character_top_k=10, beam_width=100,
                     vocabulary_size=50000, frequency_weight=0.1, lm_weight=0.15, limit=20):

        if method == "cnn":
            return self.predict_word_cnn(
                model,
                word,
                device,
                character_top_k,
                beam_width,
            )[:limit]

        if method == "wordfreq":
            candidates = self.predict_word_wordfreq(
                model,
                word,
                device,
                character_top_k,
                vocabulary_size,
                frequency_weight,
                limit,
            )
            return self.restore_size_based_case(candidates, word)

        if method == "hybrid":
            return self.predict_word_hybrid(
                model,
                word,
                device,
                character_top_k,
                beam_width,
                frequency_weight,
                lm_weight,
                limit,
            )

        if method == "wordfreq_hybrid":
            candidates = self.predict_word_wordfreq_hybrid(
                model,
                word,
                device,
                character_top_k,
                vocabulary_size,
                frequency_weight,
                lm_weight,
                limit,
            )
            return self.restore_size_based_case(candidates, word)

        raise ValueError(
            f"Unknown prediction method: {method}"
        )

    # Compatibility

    def predict_word_candidates(self, model, word, device, k=20, beam_width=100):
        return self.predict_word(
            model,
            word,
            device,
            method="cnn",
            character_top_k=k,
            beam_width=beam_width,
            limit=k,
        )

    def word_frequency_score(self, word):
        return self.wordfreq_score(word)

    def language_score(self, word):
        return self.lm_score(word)

    # Segmented text

    def predict(self, segmented, model, device, method="hybrid", **kwargs):
        result = []

        for line in segmented:
            line_result = []

            for word in line:
                candidates = self.predict_word(
                    model,
                    word,
                    device,
                    method=method,
                    **kwargs,
                )

                line_result.append(
                    candidates[0][0]
                    if candidates
                    else ""
                )

            result.append(line_result)

        return result
