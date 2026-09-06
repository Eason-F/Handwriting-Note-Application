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
    
    def dilate_character(self, image: Image.Image) -> Image.Image:
        image_array = np.asarray(image).copy()
        image_array = np.where(image_array < 170, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(image_array, kernel, iterations=1)
        return Image.fromarray(255 - dilated)

    def predict_character(self, model: torch.nn.Module, character: Image.Image, device: torch.device) -> tuple[int, float]:
        character = self.dilate_character(character)
        # character.show()
        tensor = prepare_symbol_image(character).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        return prediction.item(), confidence.item()