import unittest

import numpy as np
import torch
from PIL import Image

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
    def test_sparse_tall_strokes_remain_in_their_line(self):
        page = np.zeros((65, 80), dtype=bool)
        page[5:7, 5:14] = True
        page[5:36, 9] = True
        page[14:25, 10:70] = True
        page[46:58, 10:70] = True

        regions = LineSegmenter().find_regions(page)

        self.assertEqual(regions, [(5, 36), (46, 58)])

    def test_detached_dot_is_joined_to_the_nearby_text_line(self):
        page = np.zeros((65, 80), dtype=bool)
        page[8:10, 18:20] = True
        page[14:25, 10:70] = True
        page[46:58, 10:70] = True

        regions = LineSegmenter().find_regions(page)

        self.assertEqual(regions, [(8, 25), (46, 58)])

    def test_resolution_normalization_matches_replicated_ink(self):
        page = np.zeros((80, 180), dtype=bool)
        for left, height in ((10, 20), (35, 28), (60, 24), (95, 18)):
            page[10:10 + height, left:left + 8] = True
        expected = SegmentationPipeline.normalize_resolution(page)
        for factor in (2, 3):
            actual = SegmentationPipeline.normalize_resolution(self.scale(page, factor))
            np.testing.assert_array_equal(actual, expected)

    def test_resolution_normalization_preserves_blank_input(self):
        blank = np.zeros((20, 30), dtype=bool)
        np.testing.assert_array_equal(SegmentationPipeline.normalize_resolution(blank), blank)

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
    def test_strict_metric_preserves_case_and_punctuation(self):
        self.assertEqual(error_counts('Pa!', 'pa', strict=True), (2, 3))
        self.assertEqual(error_counts('Pa!', 'pa'), (0, 2))

    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_text('Hello, WORLD!'), 'hello world')

    def test_edit_distance_handles_insertions_and_substitutions(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)

    def test_word_error_counts(self):
        self.assertEqual(error_counts('one two three', 'one too three', words=True), (1, 3))


class AmbiguousCharacterTests(unittest.TestCase):
    def test_cnn_decoder_preserves_model_classes_without_geometry_or_case_merging(self):
        pipeline = PredictionPipeline.__new__(PredictionPipeline)
        pipeline.characters = ['I', 'i', 'l']
        image = Image.new('L', (32, 32), 255)
        image.info['geometry'] = {'components': 1}

        def model(batch):
            return torch.tensor([[0.8, 0.1, 0.1]]).log().expand(len(batch), -1)

        candidates = pipeline.predict_word_cnn(model, [image], 'cpu')
        self.assertEqual(candidates[0][0], 'I')
        self.assertAlmostEqual(float(np.exp(candidates[0][1])), 0.8, places=6)

    @staticmethod
    def glyph(top, bottom):
        image = Image.new('L', (32, 32), 255)
        image.info['geometry'] = {
            'height_ratio': bottom - top, 'top_ratio': top, 'bottom_ratio': bottom,
        }
        return image

    def test_capital_p_extends_upward_but_lowercase_p_descends(self):
        peers = [self.glyph(0.4, 0.8), self.glyph(0.4, 0.8)]
        capital = [self.glyph(0.15, 0.8)] + peers
        lowercase = [self.glyph(0.4, 1.0)] + peers
        self.assertEqual(PredictionPipeline.restore_size_based_case([('pan', 0)], capital), [('Pan', 0)])
        self.assertEqual(PredictionPipeline.restore_size_based_case([('pan', 0)], lowercase), [('pan', 0)])

    def test_tall_lowercase_ascender_is_not_capitalized_by_size(self):
        glyphs = [self.glyph(0.1, 0.8), self.glyph(0.4, 0.8), self.glyph(0.4, 0.8)]
        self.assertEqual(PredictionPipeline.restore_size_based_case([('has', 0)], glyphs), [('has', 0)])

    def test_missing_geometry_does_not_invent_case(self):
        glyphs = [Image.new('L', (32, 32), 255) for _ in range(3)]
        self.assertEqual(PredictionPipeline.restore_size_based_case([('pan', 0)], glyphs), [('pan', 0)])

    def test_case_evidence_is_summed_regardless_of_class_order(self):
        for raw in ({'P': 0.6, 'p': 0.2}, {'p': 0.2, 'P': 0.6}):
            candidates = PredictionPipeline.merge_class_evidence(raw)
            self.assertAlmostEqual(candidates['p'], 0.8)
            self.assertAlmostEqual(candidates['P'], 0.6)

    def test_visual_alternative_survives_a_lower_ranked_letter(self):
        for raw in ({'0': 0.8, 'o': 0.1}, {'o': 0.1, '0': 0.8}):
            candidates = PredictionPipeline.merge_class_evidence(raw)
            self.assertAlmostEqual(candidates['o'], 0.36)

    def test_dot_evidence_prefers_i_over_l(self):
        candidates = {'i': 0.25, 'l': 0.45, '1': 0.20}
        PredictionPipeline.apply_geometry_to_ambiguous_classes(candidates, {'components': 2})
        self.assertGreater(candidates['i'], candidates['l'])

    def test_undotted_tall_stroke_prefers_l_over_i(self):
        candidates = {'i': 0.40, 'l': 0.30, '1': 0.20}
        PredictionPipeline.apply_geometry_to_ambiguous_classes(candidates, {'components': 1})
        self.assertGreater(candidates['l'], candidates['i'])


class SegmentationHypothesisTests(unittest.TestCase):
    def test_mismatched_layouts_do_not_silently_drop_words(self):
        pipeline = PredictionPipeline.__new__(PredictionPipeline)
        for hypotheses in ([[[]], []], ([[['a']]], [[['a'], ['b']]])):
            with self.assertRaisesRegex(ValueError, 'same line and word layout'):
                pipeline.predict_best_segmentation(hypotheses, None, None)

    def test_missing_dictionary_entry_falls_back_to_cnn(self):
        class StubPipeline(PredictionPipeline):
            def __init__(self):
                pass

            def predict_word(self, model, characters, device, method, **kwargs):
                return [('123', -1.0)] if method == 'cnn' else []

        result, selected = StubPipeline().predict_best_segmentation([[[['1', '2', '3']]]], None, None)
        self.assertEqual(result, [['123']])
        self.assertEqual(selected, [[['1', '2', '3']]])

    def test_best_normalized_word_score_selects_the_segmentation(self):
        class StubPipeline(PredictionPipeline):
            def __init__(self):
                pass

            def predict_word(self, model, characters, device, **kwargs):
                return [(''.join(characters), {1: -4.0, 2: -5.0}[len(characters)])]

        hypotheses = [[[['a']]], [[['a', 'b']]]]
        result, selected = StubPipeline().predict_best_segmentation(hypotheses, None, None)

        self.assertEqual(result, [['ab']])
        self.assertEqual(selected, [[['a', 'b']]])


if __name__ == '__main__':
    unittest.main()
