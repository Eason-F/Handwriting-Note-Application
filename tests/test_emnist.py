import unittest

import numpy as np

from hasy_cnn.data import (
    EMNISTDataset,
    EMNIST_BYCLASS_CHARACTERS,
    SymbolInfo,
    add_emnist_characters,
)


class FakeSource:
    def __init__(self):
        image = np.zeros((28, 28, 1), dtype=np.uint8)
        image[4:24, 12:16, 0] = 255
        self.examples = [{"image": image, "label": 61}]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class EMNISTAdapterTests(unittest.TestCase):
    def test_missing_lowercase_t_is_added(self):
        original = [SymbolInfo(1, "A"), SymbolInfo(2, "0")]
        extended = add_emnist_characters(original)
        labels = {symbol.latex for symbol in extended}
        self.assertTrue(set(EMNIST_BYCLASS_CHARACTERS).issubset(labels))

    def test_tfds_sample_becomes_hasy_shaped_tensor(self):
        symbols = add_emnist_characters([SymbolInfo(1, "A")])
        dataset = EMNISTDataset(FakeSource(), symbols)
        image, class_index = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 32, 32))
        self.assertEqual(symbols[class_index].latex, "z")
        self.assertEqual(image[0, 0, 0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
