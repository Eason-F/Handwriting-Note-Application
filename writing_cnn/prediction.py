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

        distance = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
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
                ink = cv2.dilate(ink.astype(np.uint8), kernel, 1).astype(bool)
            else:
                eroded = cv2.erode(ink.astype(np.uint8), kernel, 1)

                if not np.any(eroded):
                    break

                ink = eroded.astype(bool)

        output = np.full_like(image_array, 255)
        output[ink] = 0

        return Image.fromarray(output)

    def predict_character(self, model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
        character = self.normalize_stroke_thickness(character)
        tensor = prepare_symbol_image(character).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()