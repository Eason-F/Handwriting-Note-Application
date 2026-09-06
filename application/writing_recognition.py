import time
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = None

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from writing_cnn.data import EMNIST_BYCLASS_CHARACTERS
from writing_cnn.model import WritingCNN
from writing_cnn.prediction import PredictionPipeline
from writing_cnn.segmentation import CharacterSegmenter, ConjoinedCharacterSegmenter, LineSegmenter, SegmentationPipeline, WordSegmenter

from .theme import CHECKPOINT_PATH


@dataclass
class RecognitionResult:
    text: str
    elapsed_ms: float
    lines: int
    words: int
    characters: int


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class RecognitionWorker(QRunnable):
    def __init__(self, recognizer, image):
        super().__init__()
        self.recognizer = recognizer
        self.image = image
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.finished.emit(self.recognizer.recognize(self.image))
        except Exception as exc:
            self.signals.error.emit(str(exc))


class HandwritingRecognizer:
    def __init__(self):
        self.device = torch.device('mps' if torch and torch.backends.mps.is_available() else 'cpu') if torch else None
        self.model = None
        self.pipeline = None
        self.segmentation = None
        self.error = None
        if torch is None:
            self.error = 'PyTorch is not installed.'
            return
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.model = WritingCNN(checkpoint['number_of_classes'])
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
            self.pipeline = PredictionPipeline(list(EMNIST_BYCLASS_CHARACTERS))
            self.segmentation = SegmentationPipeline(
                LineSegmenter(min_ink_pixels=8, max_internal_gap=2, min_line_height=10),
                WordSegmenter(), CharacterSegmenter(),
                ConjoinedCharacterSegmenter(width_multiplier=1.8, min_character_width=5),
            )
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self):
        return all(value is not None for value in (self.model, self.pipeline, self.segmentation))

    def recognize(self, image):
        started = time.perf_counter()
        if not self.ready:
            raise RuntimeError(f'Writing model unavailable: {self.error or "unknown error"}')
        segmented = self.segmentation.segment(image.convert('L'), self.model, self.device)
        lines, words, chars = [], 0, 0
        for line in segmented:
            final_words = []
            for word in line:
                words += 1
                chars += len(word)
                candidates = self.pipeline.predict_word_candidates(self.model, word, self.device, k=5, beam_width=10)
                reranked = self.pipeline.rerank_word_candidates(candidates, frequency_weight=0.20, language_weight=0.35)
                final_words.append(reranked[0][0] if reranked else candidates[0][0] if candidates else '')
            lines.append(' '.join(final_words))
        return RecognitionResult('\n'.join(lines), (time.perf_counter() - started) * 1000, len(lines), words, chars)
