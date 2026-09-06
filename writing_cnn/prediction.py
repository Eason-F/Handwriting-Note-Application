from collections import Counter

from PIL import Image
import cv2
import numpy as np
import torch
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

        return float(score)


# Prediction pipeline

class PredictionPipeline:
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

        distance = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
        values = distance[distance > 0]

        return float(np.median(values) * 2) if len(values) else 0.0

    @staticmethod
    def normalize_stroke_thickness(image, target_width=3.0, tolerance=0.5):
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
                ink = cv2.dilate(ink.astype(np.uint8), kernel, 1).astype(bool)
                continue

            eroded = cv2.erode(ink.astype(np.uint8), kernel, 1)

            if not np.any(eroded):
                break

            ink = eroded.astype(bool)

        output = np.full_like(image_array, 255)
        output[ink] = 0

        return Image.fromarray(output)

    @staticmethod
    def prepare_character(character):
        return prepare_symbol_image(
            PredictionPipeline.normalize_stroke_thickness(character)
        )

    # Character prediction

    @staticmethod
    def predict(model, character, device):
        character = PredictionPipeline.prepare_character(character)
        tensor = character.unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()

    @staticmethod
    def predict_topk(model, character, device, k=10):
        tensor = PredictionPipeline.prepare_character(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)

        k = min(k, probabilities.shape[1])
        confidence, prediction = torch.topk(probabilities, k, dim=1)

        return [(p.item(), c.item()) for p, c in zip(prediction[0], confidence[0])]

    def predict_character(self, model, character, device):
        prediction, confidence = self.predict(model, character, device)
        return self.characters[prediction], confidence

    def predict_character_candidates(self, model, character, device, k=10):
        return [(self.characters[p], c) for p, c in self.predict_topk(model, character, device, k)]

    # CNN word prediction

    def predict_word_cnn(self, model, word, device, character_top_k=10, beam_width=100):
        beams = [("", 0.0)]

        for character in word:
            candidates = self.predict_character_candidates(model, character, device, character_top_k)
            next_beams = []

            for text, score in beams:
                for prediction, probability in candidates:
                    next_beams.append((text + prediction, score + np.log(max(probability, 1e-8))))

            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:beam_width]

        return beams

    # WordFreq

    @staticmethod
    def normalize_word(word):
        return "".join(c for c in word.lower() if c.isalpha())

    @staticmethod
    def wordfreq_score(word):
        word = PredictionPipeline.normalize_word(word)
        return float(zipf_frequency(word, "en", wordlist="best")) if word else 0.0

    def _load_vocabulary(self, vocabulary_size=50000):
        if self._vocabulary is not None:
            return

        self._vocabulary = {
            word.lower()
            for word in top_n_list("en", vocabulary_size, wordlist="best")
            if word.isalpha()
        }

        self._vocabulary_by_length = {}

        for word in self._vocabulary:
            self._vocabulary_by_length.setdefault(len(word), []).append(word)

    def score_word_against_cnn(self, word, character_candidates):
        score = 0.0

        for character, candidates in zip(word, character_candidates):
            probabilities = dict(candidates)
            score += np.log(max(probabilities.get(character, 1e-8), 1e-8))

        return float(score)

    def predict_word_wordfreq(self, model, word, device, character_top_k=10, vocabulary_size=50000, frequency_weight=0.1, limit=20):
        self._load_vocabulary(vocabulary_size)

        character_candidates = [
            self.predict_character_candidates(model, character, device, character_top_k)
            for character in word
        ]

        candidates = []

        for vocabulary_word in self._vocabulary_by_length.get(len(word), []):
            cnn_score = self.score_word_against_cnn(vocabulary_word, character_candidates)
            frequency = self.wordfreq_score(vocabulary_word)
            candidates.append((vocabulary_word, cnn_score + frequency * frequency_weight))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]

    # Language model

    def lm_score(self, word):
        word = self.normalize_word(word)
        return self.language_model.score(word) if word else -20.0

    # Combined scoring

    def combined_score(self, word, cnn_score=0.0, frequency_weight=0.1, lm_weight=0.1):
        word = self.normalize_word(word)

        return float(
            cnn_score
            + frequency_weight * self.wordfreq_score(word)
            + lm_weight * self.lm_score(word)
        )

    def rerank_word_candidates(self, candidates, frequency_weight=0.1, lm_weight=0.1):
        reranked = [
            (
                word,
                self.combined_score(word, cnn_score, frequency_weight, lm_weight),
            )
            for word, cnn_score in candidates
        ]

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # Word prediction

    def predict_word(self, model, word, device, method="hybrid", character_top_k=10, beam_width=100,
                     vocabulary_size=50000, frequency_weight=0.1, lm_weight=0.1, limit=20):

        if method == "cnn":
            return self.predict_word_cnn(model, word, device, character_top_k, beam_width)[:limit]

        if method == "wordfreq":
            return self.predict_word_wordfreq(
                model,
                word,
                device,
                character_top_k,
                vocabulary_size,
                frequency_weight,
                limit,
            )

        if method == "hybrid":
            candidates = self.predict_word_cnn(
                model,
                word,
                device,
                character_top_k,
                beam_width,
            )
            return self.rerank_word_candidates(
                candidates,
                frequency_weight,
                lm_weight,
            )[:limit]

        if method == "wordfreq_hybrid":
            candidates = self.predict_word_wordfreq(
                model,
                word,
                device,
                character_top_k,
                vocabulary_size,
                0.0,
                beam_width,
            )
            return self.rerank_word_candidates(
                candidates,
                frequency_weight,
                lm_weight,
            )[:limit]

        raise ValueError(
            f"Unknown prediction method: {method}"
        )

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
                    method,
                    **kwargs,
                )
                line_result.append(candidates[0][0] if candidates else "")

            result.append(line_result)

        return result