import unittest

import torch
from PIL import Image, ImageDraw

from hasy_cnn.images import crop_and_center_drawing, prepare_symbol_image
from hasy_cnn.model import SymbolCNN


class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        model = SymbolCNN(369)
        output = model(torch.zeros(4, 1, 32, 32))
        self.assertEqual(tuple(output.shape), (4, 369))

    def test_preprocessing_shape_and_range(self):
        image = Image.new("L", (32, 32), 255)
        tensor = prepare_symbol_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 32, 32))
        self.assertGreaterEqual(tensor.min().item(), -1.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_crop_and_center(self):
        image = Image.new("L", (300, 300), 255)
        draw = ImageDraw.Draw(image)
        draw.line((140, 60, 140, 240), fill=0, width=15)
        result = crop_and_center_drawing(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.size, (32, 32))


if __name__ == "__main__":
    unittest.main()

