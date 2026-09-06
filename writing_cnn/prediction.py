from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

from math_cnn.images import prepare_symbol_image
from wordfreq import top_n_list, zipf_frequency


class CharacterLanguageModel:
    def __init__(self, vocabulary_size=50000, n=4, smoothing=0.1):
        self.n = n
        self.smoothing = smoothing
        self.vocabulary_size = vocabulary_size
        self.counts = Counter()
        self.contexts = Counter()
        self._build()

    def _build(self):
        for word in top_n_list("en", self.vocabulary_size, wordlist="best"):
            word = "".join(c for c in word.lower() if c.isalpha())

            if not word:
                continue

            weight = max(
                1,
                int(
                    10 ** max(
                        zipf_frequency(word, "en", wordlist="best") - 3,
                        0,
                    )
                ),
            )

            text = "^" * (self.n - 1) + word + "$"

            for i in range(len(text) - self.n + 1):
                context = text[i:i + self.n - 1]
                character = text[i + self.n - 1]

                self.counts[(context, character)] += weight
                self.contexts[context] += weight

    def score(self, word):
        word = "".join(c for c in word.lower() if c.isalpha())

        if not word:
            return -20.0

        text = "^" * (self.n - 1) + word + "$"
        score = 0.0

        for i in range(len(text) - self.n + 1):
            context = text[i:i + self.n - 1]
            character = text[i + self.n - 1]

            numerator = self.counts[(context, character)] + self.smoothing
            denominator = self.contexts[context] + self.smoothing * 27

            score += np.log(numerator / denominator)

        return score


class PredictionPipeline:
    def __init__(self, characters: list[str]):
        self.characters = characters
        self._vocabulary = None
        self._vocabulary_by_length = {}
        self.language_model = CharacterLanguageModel()

    def predict(self, segmented: list[list[list[Image.Image]]], model: torch.nn.Module, device: torch.device) -> list[list[str]]:
        result = []

        for line in segmented:
            line_result = []

            for word in line:
                text = ""

                for character in word:
                    prediction, _ = self.predict_character(
                        model,
                        character,
                        device,
                    )
                    text += self.characters[prediction]

                line_result.append(text)

            result.append(line_result)

        return result

    @staticmethod
    def estimate_stroke_width(ink: np.ndarray) -> float:
        if not np.any(ink):
            return 0.0

        distance = cv2.distanceTransform(
            ink.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        values = distance[distance > 0]

        return float(np.median(values) * 2)

    @staticmethod
    def normalize_stroke_thickness(image: Image.Image, target_width: float = 3.0, tolerance: float = 0.5) -> Image.Image:
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

        return Image.fromarray(output)

    @staticmethod
    def prepare_character(character: Image.Image) -> torch.Tensor:
        character = PredictionPipeline.normalize_stroke_thickness(character)
        return prepare_symbol_image(character)

    @staticmethod
    def predict_character(model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
        tensor = PredictionPipeline.prepare_character(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        confidence, prediction = probabilities.max(dim=1)

        return prediction.item(), confidence.item()

    @staticmethod
    def predict_character_topk(model: torch.nn.Module, character: Image.Image, device: torch.device, k: int = 10) -> list[tuple[int, float]]:
        tensor = PredictionPipeline.prepare_character(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

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

    def predict_character_candidates(self, model: torch.nn.Module, character: Image.Image, device: torch.device, k: int = 10) -> list[tuple[str, float]]:
        return [
            (
                self.characters[prediction],
                confidence,
            )
            for prediction, confidence in self.predict_character_topk(
                model,
                character,
                device,
                k,
            )
        ]

    def predict_word_candidates(self, model: torch.nn.Module, word: list[Image.Image], device: torch.device, k: int = 10, beam_width: int = 100) -> list[tuple[str, float]]:
        beams = [("", 0.0)]

        for character in word:
            candidates = self.predict_character_candidates(
                model,
                character,
                device,
                k,
            )

            next_beams = []

            for text, score in beams:
                for prediction, confidence in candidates:
                    confidence = max(confidence, 1e-6)

                    next_beams.append(
                        (
                            text + prediction,
                            score + float(np.log(confidence)),
                        )
                    )

            next_beams.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            beams = next_beams[:beam_width]

        return beams

    @staticmethod
    def normalize_word(word: str) -> str:
        return "".join(
            character
            for character in word.lower()
            if character.isalpha()
        )

    @staticmethod
    def word_edit_distance(source: str, target: str) -> int:
        previous = list(range(len(target) + 1))

        for i, source_character in enumerate(source, start=1):
            current = [i]

            for j, target_character in enumerate(target, start=1):
                current.append(
                    min(
                        current[j - 1] + 1,
                        previous[j] + 1,
                        previous[j - 1]
                        + (source_character != target_character),
                    )
                )

            previous = current

        return previous[-1]

    @staticmethod
    def word_frequency_score(word: str) -> float:
        word = PredictionPipeline.normalize_word(word)

        if not word:
            return 0.0

        return zipf_frequency(
            word,
            "en",
            wordlist="best",
        )

    def language_score(self, word: str) -> float:
        return self.language_model.score(word)

    def _load_vocabulary(self, vocabulary_size: int = 50000):
        if self._vocabulary is not None:
            return

        words = top_n_list(
            "en",
            vocabulary_size,
            wordlist="best",
        )

        self._vocabulary = {
            word.lower()
            for word in words
            if word.isalpha()
        }

        self._vocabulary_by_length = {}

        for word in self._vocabulary:
            self._vocabulary_by_length.setdefault(
                len(word),
                [],
            ).append(word)

    def score_vocabulary_word(self, word: str, character_candidates: list[list[tuple[str, float]]], frequency_weight: float = 0.10) -> float:
        score = 0.0

        for character, candidates in zip(
            word,
            character_candidates,
        ):
            probabilities = dict(candidates)
            probability = max(
                probabilities.get(character, 1e-6),
                1e-6,
            )

            score += np.log(probability)

        return (
            score
            + self.word_frequency_score(word)
            * frequency_weight
        )

    def predict_word_vocabulary(self, model: torch.nn.Module, word: list[Image.Image], device: torch.device, k: int = 10, vocabulary_size: int = 50000, frequency_weight: float = 0.10) -> list[tuple[str, float]]:
        self._load_vocabulary(vocabulary_size)

        character_candidates = [
            self.predict_character_candidates(
                model,
                character,
                device,
                k,
            )
            for character in word
        ]

        candidates = []

        for vocabulary_word in self._vocabulary_by_length.get(
            len(word),
            [],
        ):
            score = self.score_vocabulary_word(
                vocabulary_word,
                character_candidates,
                frequency_weight,
            )

            candidates.append(
                (
                    vocabulary_word,
                    score,
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return candidates[:20]

    def rerank_word_candidates(self, candidates: list[tuple[str, float]], frequency_weight: float = 0.10, language_weight: float = 0.10) -> list[tuple[str, float]]:
        reranked = []

        for word, cnn_score in candidates:
            normalized = self.normalize_word(word)
            frequency = self.word_frequency_score(normalized)
            language = self.language_score(normalized)

            score = (
                cnn_score
                + frequency * frequency_weight
                + language * language_weight
            )

            reranked.append(
                (
                    word,
                    score,
                )
            )

        reranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked