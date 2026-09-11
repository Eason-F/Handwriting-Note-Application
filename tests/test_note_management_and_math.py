import tempfile
import unittest
from pathlib import Path

from application.calculator import MathEvaluationError, MathEvaluator
from application.notes import NoteDocument, NoteError, NoteManager


class MathEvaluatorTests(unittest.TestCase):
    def test_symbolic_operations_and_implicit_multiplication(self):
        evaluator = MathEvaluator()
        self.assertEqual(evaluator.evaluate('2x + 3x').result, '5*x')
        self.assertEqual(evaluator.evaluate('expand((x + 1)^2)').result, 'x**2 + 2*x + 1')
        self.assertEqual(evaluator.evaluate('factor(x^2 - 1)').result, '(x - 1)*(x + 1)')
        self.assertEqual(evaluator.evaluate('solve(x^2 - 4, x)').result, '[-2, 2]')

    def test_assignments_persist_in_one_context(self):
        evaluator = MathEvaluator()
        evaluator.evaluate('x = 5')
        evaluator.evaluate('y = x + 2')
        self.assertEqual(evaluator.evaluate('x * y').result, '35')

    def test_unicode_and_errors_are_handled(self):
        evaluator = MathEvaluator()
        self.assertEqual(evaluator.evaluate('2 × 3').result, '6')
        with self.assertRaises(MathEvaluationError):
            evaluator.evaluate('1 ÷ 0')
        with self.assertRaises(MathEvaluationError):
            evaluator.evaluate('unknown(2)')

    def test_equation_query_solves_for_requested_variable(self):
        evaluator = MathEvaluator()
        self.assertEqual(evaluator.evaluate('5x = 10 x=?').result, 'x = 2')
        evaluator.evaluate('x = 99')
        self.assertEqual(evaluator.evaluate('5x = 10 x=?').result, 'x = 2')


class NoteManagerTests(unittest.TestCase):
    def test_document_round_trip_includes_raw_strokes(self):
        with tempfile.TemporaryDirectory() as root:
            manager = NoteManager(root)
            document = manager.create_note('Class notes')
            document.text = 'hello'
            document.strokes = [{'tool': 'pen', 'width': 5, 'points': [[1, 2], [3, 4]]}]
            document.dirty = True
            manager.save_note(document)
            reopened = manager.open_note(document.path)
            self.assertEqual(reopened.text, 'hello')
            self.assertEqual(reopened.strokes, document.strokes)
            self.assertFalse(reopened.dirty)

    def test_duplicate_names_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            manager = NoteManager(root)
            first = manager.create_note('Note')
            second = manager.create_note('Note')
            self.assertNotEqual(first.path, second.path)
            with self.assertRaises(NoteError):
                manager.save_note(NoteDocument(text='replacement'), first.path)


if __name__ == '__main__':
    unittest.main()
