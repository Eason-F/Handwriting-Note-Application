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
from writing_cnn.segmentation import (
    CharacterSegmenter,
    ConjoinedCharacterSegmenter,
    LineSegmenter,
    SegmentationPipeline,
    WordSegmenter,
)

from .theme import CHECKPOINT_PATH


@dataclass
class RecognitionResult:
    text: str
    elapsed_ms: float
    lines: int
    words: int
    characters: int
    segmentation_ms: float
    raw_text: str = ''


class WorkerSignals(QObject):
    finished = Signal(object, object, object, object, bool)
    error = Signal(str, object, object, object, bool)


class RecognitionWorker(QRunnable):
    def __init__(self, recognizer, image, dialog=None, canvas=None, info=None, automatic=False):
        super().__init__()
        self.setAutoDelete(True)
        self.recognizer = recognizer
        self.image = image
        self.dialog = dialog
        self.canvas = canvas
        self.info = info
        self.automatic = automatic
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.recognizer.recognize(self.image)
            self.signals.finished.emit(result, self.dialog, self.canvas, self.info, self.automatic)
        except Exception as exc:
            self.signals.error.emit(f'{type(exc).__name__}: {exc}', self.dialog, self.canvas, self.info, self.automatic)


class HandwritingRecognizer:
    def __init__(self):
        if torch is None:
            self.device = None
            self.model = None
            self.pipeline = None
            self.segmentation = None
            self.error = 'PyTorch is not installed.'
            return
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.pipeline = None
        self.segmentation = None
        self.error = None
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.model = WritingCNN(checkpoint['number_of_classes'])
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
            self.pipeline = PredictionPipeline(list(EMNIST_BYCLASS_CHARACTERS))
            self.segmentation = SegmentationPipeline(
                LineSegmenter(),
                WordSegmenter(),
                CharacterSegmenter(),
                ConjoinedCharacterSegmenter(width_multiplier=1.8, min_character_width=5),
            )
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    @property
    def ready(self):
        return self.model is not None and self.pipeline is not None and self.segmentation is not None

    def segment(self, image):
        if not self.ready:
            raise RuntimeError(f'Writing model unavailable: {self.error or "unknown error"}')
        return self.segmentation.segment(image.convert('L'), self.model, self.device)

    def recognize(self, image):
        started = time.perf_counter()
        segmentation_started = time.perf_counter()
        segmented = self.segment(image)
        segmentation_ms = (time.perf_counter() - segmentation_started) * 1000

        final_lines = self.pipeline.predict(
            segmented,
            self.model,
            self.device,
            method='wordfreq_hybrid',
            character_top_k=8,
            beam_width=100,
            frequency_weight=0.50,
            lm_weight=0.5,
        )
        raw_lines = self.pipeline.predict(segmented, self.model, self.device, method='cnn')
        words = sum(len(line) for line in final_lines)
        characters = sum(len(word) for line in segmented for word in line)

        return RecognitionResult(
            text='\n'.join([' '.join(word) for word in final_lines]),
            raw_text='\n'.join([' '.join(word) for word in raw_lines]),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            lines=len(segmented),
            words=words,
            characters=characters,
            segmentation_ms=segmentation_ms,
        )
