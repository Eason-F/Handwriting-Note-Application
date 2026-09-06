from PIL import Image

import torch
import numpy as np
import cv2

from math_cnn.images import prepare_symbol_image


class PredictionPipeline:
    def __init__(self, characters: list[str]):
        self.characters = characters

    def predict(self, segmented: list[list[list[Image.Image]]], model: torch.nn.Module, device: torch.device) -> list[list[str]]:
        result = []

        for line in segmented:
            line_result = []

            for word in line:
                predictions = []

                for character in word:
                    prediction, _ = self.predict_character(model, character, device)
                    predictions.append(self.characters[prediction])

                line_result.append("".join(predictions))

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
            else:
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
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        return prediction.item(), confidence.item()

    @staticmethod
    def predict_character_topk(model: torch.nn.Module, character: Image.Image, device: torch.device, k: int = 5) -> list[tuple[int, float]]:
        tensor = PredictionPipeline.prepare_character(character).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

        k = min(k, probabilities.shape[1])

        confidences, predictions = torch.topk(
            probabilities,
            k=k,
            dim=1,
        )

        return [
            (prediction.item(), confidence.item())
            for prediction, confidence in zip(
                predictions[0],
                confidences[0],
            )
        ]

    def predict_character_candidates(self, model: torch.nn.Module, character: Image.Image, device: torch.device, k: int = 5) -> list[tuple[str, float]]:
        return [
            (self.characters[prediction], confidence)
            for prediction, confidence in self.predict_character_topk(
                model,
                character,
                device,
                k,
            )
        ]

    def predict_word_candidates(self, model: torch.nn.Module, word: list[Image.Image], device: torch.device, k: int = 5, beam_width: int = 10) -> list[tuple[str, float]]:
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
                for character_name, confidence in candidates:
                    if confidence <= 0:
                        continue

                    next_beams.append(
                        (
                            text + character_name,
                            score + float(np.log(confidence)),
                        )
                    )

            next_beams.sort(
                key=lambda candidate: candidate[1],
                reverse=True,
            )

            beams = next_beams[:beam_width]

        return beams