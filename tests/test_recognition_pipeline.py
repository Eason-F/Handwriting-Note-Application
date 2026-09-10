import unittest

import numpy as np

from writing_cnn.segmentation import CharacterSegmenter, LineSegmenter, SegmentationPipeline, WordSegmenter
from writing_cnn.benchmark import edit_distance, error_counts, normalize_text
from writing_cnn.prediction import PredictionPipeline


class InkCroppingTests(unittest.TestCase):
    def test_writing_in_first_column_is_not_treated_as_blank(self):
        binary = np.zeros((8, 8), dtype=bool)
        binary[2:6, 0] = True

        cropped = SegmentationPipeline.find_writing_region(binary)

        self.assertEqual(cropped.shape, (4, 1))
        self.assertTrue(cropped.all())

    def test_line_crop_accepts_ink_in_first_column(self):
        binary = np.zeros((8, 3), dtype=bool)
        binary[1:7, 0] = True

        cropped = LineSegmenter.crop_to_ink(binary)

        self.assertEqual(cropped.shape, (6, 1))

    def test_word_crop_accepts_ink_in_first_column(self):
        binary = np.zeros((5, 4), dtype=bool)
        binary[:, 0] = True

        cropped = WordSegmenter.crop_to_ink(binary)

        self.assertEqual(cropped.shape, (5, 1))

    def test_character_crop_keeps_first_column_character(self):
        binary = np.zeros((6, 5), dtype=bool)
        binary[1:5, 0] = True

        characters = CharacterSegmenter().crop(binary, [(0, 1)])

        self.assertEqual(len(characters), 1)

    def test_blank_input_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "No handwriting detected"):
            SegmentationPipeline.find_writing_region(np.zeros((5, 5), dtype=bool))


class ScaleAdaptiveSegmentationTests(unittest.TestCase):
    @staticmethod
    def scale(binary, factor):
        return np.repeat(np.repeat(binary, factor, axis=0), factor, axis=1)

    def test_word_boundaries_are_stable_across_integer_scales(self):
        line = np.zeros((20, 76), dtype=bool)
        for start, end in ((2, 8), (10, 17), (27, 34), (36, 42), (52, 59), (61, 70)):
            line[3:17, start:end] = True

        for factor in (1, 2, 3):
            scaled = self.scale(line, factor)
            regions = WordSegmenter().find_regions(scaled)
            self.assertEqual(len(regions), 3)

    def test_line_boundaries_are_stable_across_integer_scales(self):
        page = np.zeros((58, 90), dtype=bool)
        page[5:20, 8:82] = True
        page[36:51, 8:82] = True

        for factor in (1, 2, 3):
            scaled = self.scale(page, factor)
            regions = LineSegmenter().find_regions(scaled)
            self.assertEqual(len(regions), 2)


class RecognitionMetricTests(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_text('Hello, WORLD!'), 'hello world')

    def test_edit_distance_handles_insertions_and_substitutions(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)

    def test_word_error_counts(self):
        self.assertEqual(error_counts('one two three', 'one too three', words=True), (1, 3))


class AmbiguousCharacterTests(unittest.TestCase):
    def test_dot_evidence_prefers_i_over_l(self):
        candidates = {'i': 0.25, 'l': 0.45, '1': 0.20}
        PredictionPipeline.apply_geometry_to_ambiguous_classes(candidates, {'components': 2})
        self.assertGreater(candidates['i'], candidates['l'])

    def test_undotted_tall_stroke_prefers_l_over_i(self):
        candidates = {'i': 0.40, 'l': 0.30, '1': 0.20}
        PredictionPipeline.apply_geometry_to_ambiguous_classes(candidates, {'components': 1})
        self.assertGreater(candidates['l'], candidates['i'])


if __name__ == '__main__':
    unittest.main()
