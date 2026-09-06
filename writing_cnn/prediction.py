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

        distances = distance[ink > 0]

        if len(distances) == 0:
            return 0.0

        return float(np.median(distances) * 2.0)
    
    def normalize_stroke_thickness(self, image: Image.Image, target_width: float = 2.25, tolerance: float = 0.36) -> Image.Image:
        image_array = np.asarray(image.convert("L"))
        ink = image_array < 128

        if not np.any(ink):
            return image

        width = self.estimate_stroke_width(ink)
        print(f"Stroke width: {width:.2f}px")

        kernel = np.ones((3, 3), dtype=np.uint8)

        for _ in range(2):
            if target_width - tolerance <= width <= target_width + tolerance:
                break

            if width < target_width:
                ink = cv2.dilate(ink.astype(np.uint8), kernel, iterations=1).astype(bool)
            else:
                eroded = cv2.erode(ink.astype(np.uint8), kernel, iterations=1).astype(bool)

                if not np.any(eroded):
                    break

                ink = eroded

            width = self.estimate_stroke_width(ink)

        print(f"Stroke width after: {width:.2f}px")

        output = np.full_like(image_array, 255)
        output[ink] = 0

        return Image.fromarray(output)

    def predict_character(self, model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
        character = self.normalize_stroke_thickness(character)
        character.show()
        tensor = prepare_symbol_image(character).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()