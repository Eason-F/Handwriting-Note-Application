import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from application.main_window import MainWindow


class RecognitionResultHandlingTests(unittest.TestCase):
    @staticmethod
    def window():
        window = SimpleNamespace(
            _recognition_running=True,
            _writing_preview=None,
            _writing_preview_revision=None,
            _insert_text_at_cursor=Mock(),
            status_label=Mock(),
        )
        return window

    @staticmethod
    def result():
        return SimpleNamespace(
            text='handwritten note', lines=1, words=2, characters=15,
            segmentation_ms=20, elapsed_ms=40,
        )

    def test_automatic_result_is_a_preview_and_preserves_ink(self):
        window = self.window()
        dialog, canvas, info = Mock(), Mock(revision=7), Mock()
        dialog.isVisible.return_value = True

        info.toPlainText.return_value = ''
        MainWindow._writing_done(window, self.result(), dialog, canvas, info, True, 7)

        window._insert_text_at_cursor.assert_not_called()
        canvas.clear.assert_not_called()
        self.assertEqual(window._writing_preview.text, 'handwritten note')
        self.assertEqual(window._writing_preview_revision, 7)

    def test_manual_result_is_inserted_and_preserves_the_submitted_ink(self):
        window = self.window()
        dialog, canvas, info = Mock(), Mock(revision=7), Mock()
        dialog.isVisible.return_value = True

        info.toPlainText.return_value = ''
        MainWindow._writing_done(window, self.result(), dialog, canvas, info, False, 7)

        window._insert_text_at_cursor.assert_called_once_with('handwritten note')
        canvas.clear.assert_not_called()
        dialog.close.assert_not_called()

    def test_stale_result_does_not_insert_or_clear_newer_strokes(self):
        window = self.window()
        dialog, canvas, info = Mock(), Mock(revision=8), Mock()
        dialog.isVisible.return_value = True

        MainWindow._writing_done(window, self.result(), dialog, canvas, info, False, 7)

        window._insert_text_at_cursor.assert_not_called()
        canvas.clear.assert_not_called()
        info.setPlainText.assert_called_once()

    def test_result_from_a_closed_window_is_ignored_safely(self):
        window = self.window()
        dialog, canvas, info = Mock(), Mock(revision=7), Mock()
        info.toPlainText.return_value = ''
        MainWindow._writing_done(window, self.result(), dialog, canvas, info, True, 7)
        window._insert_text_at_cursor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
