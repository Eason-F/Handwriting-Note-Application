import tempfile
import unittest
from pathlib import Path

from writing_cnn.iam import build_manifests, official_splits
from writing_cnn.benchmark import balanced_sample, format_rate


class IAMImportTests(unittest.TestCase):
    def test_official_form_splits_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train.uttlist').write_text('a01-000u\n')
            (root / 'validation.uttlist').write_text('b01-001\n')
            (root / 'test.uttlist').write_text('c01-002\n')
            self.assertEqual(official_splits(root), {
                'a01-000u': 'train', 'b01-001': 'validation', 'c01-002': 'test',
            })
            (root / 'test.uttlist').write_text('a01-000u\n')
            with self.assertRaisesRegex(ValueError, 'multiple official splits'):
                official_splits(root)

    def test_writer_isolation_and_rejected_segmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forms, words = root / 'forms.txt', root / 'words.txt'
            forms.write_text('a01-000u 000 2 prt 7 5 52 36\na01-001 000 2 all 7 7 52 52\n')
            words.write_text(
                '# annotations\n'
                'a01-000u-00-00 ok 154 408 768 27 51 AT A\n'
                'a01-001-00-00 ok 154 408 768 27 51 NN word\n'
                'a01-001-00-01 err 154 408 768 27 51 NN rejected\n'
            )
            for form in ('a01-000u', 'a01-001'):
                image = root / 'images' / 'a01' / form / f'{form}-00-00.png'
                image.parent.mkdir(parents=True)
                image.touch()
            manifests, rejected = build_manifests(words, forms, root / 'images')
            populated = [samples for samples in manifests.values() if samples]
            self.assertEqual(len(populated), 1)
            self.assertEqual([sample['text'] for sample in populated[0]], ['A', 'word'])
            self.assertEqual(rejected, 1)
            forms.write_text('a01-000u 000\n')
            with self.assertRaisesRegex(ValueError, 'Missing writer'):
                build_manifests(words, forms, root / 'images')

    def test_empty_normalized_reference_is_reported_without_division(self):
        self.assertIn('n/a', format_rate((1, 0)))

    def test_benchmark_subset_round_robins_forms(self):
        samples = [
            {'form_id': form, 'sample_id': f'{form}-{number}'}
            for form in ('a', 'b') for number in range(3)
        ]
        chosen = balanced_sample(samples, 4)
        self.assertEqual([sample['sample_id'] for sample in chosen], ['a-0', 'b-0', 'a-1', 'b-1'])


if __name__ == '__main__':
    unittest.main()
