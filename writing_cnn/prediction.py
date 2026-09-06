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
    
    def normalize_stroke_thickness(self, image: Image.Image) -> Image.Image:
        image_array = np.asarray(image.convert("L"))
        ink = image_array < 128

        if not np.any(ink):
            return image

        ys, xs = np.where(ink)
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        character = ink[y1:y2, x1:x2]
        ink_ratio = character.mean()

        print(ink_ratio)
        if ink_ratio < 0.2:
            kernel = np.ones((3, 3), dtype=np.uint8)
            character = cv2.dilate(character.astype(np.uint8), kernel, iterations=1).astype(bool)
        elif ink_ratio > 0.48:
            kernel = np.ones((3, 3), dtype=np.uint8)
            character = cv2.erode(character.astype(np.uint8), kernel, iterations=1).astype(bool)

        output = np.full_like(image_array, 255)
        output[y1:y2, x1:x2][character] = 0

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