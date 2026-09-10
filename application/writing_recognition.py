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
    finished = Signal(object, object, object, object, bool, int)
    error = Signal(str, object, object, object, bool, int)


class RecognitionWorker(QRunnable):
    def __init__(self, recognizer, image, dialog=None, canvas=None, info=None, automatic=False, revision=0):
        super().__init__()
        self.setAutoDelete(True)
        self.recognizer = recognizer
        self.image = image
        self.dialog = dialog
        self.canvas = canvas
        self.info = info
        self.automatic = automatic
        self.revision = revision
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.recognizer.recognize(self.image)
            self.signals.finished.emit(result, self.dialog, self.canvas, self.info, self.automatic, self.revision)
        except Exception as exc:
            self.signals.error.emit(
                f'{type(exc).__name__}: {exc}', self.dialog, self.canvas,
                self.info, self.automatic, self.revision,
            )


class HandwritingRecognizer:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        if torch is None:
            self.device = None
            self.model = None
            self.pipeline = None
            self.segmentation = None
            self.segmentation_hypotheses = None
            self.error = 'PyTorch is not installed.'
            return
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.pipeline = None
        self.segmentation = None
        self.segmentation_hypotheses = None
        self.error = None
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model = WritingCNN(checkpoint['number_of_classes'])
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
            self.pipeline = PredictionPipeline(list(EMNIST_BYCLASS_CHARACTERS))
            self.segmentation = self._create_segmenter(2.2)
            self.segmentation_hypotheses = [
                self._create_segmenter(ratio).conjoined_segmenter for ratio in (1.8, 3.0)
            ]
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    @staticmethod
    def _create_segmenter(geometry_split_ratio):
        return SegmentationPipeline(
            LineSegmenter(),
            WordSegmenter(),
            CharacterSegmenter(),
            ConjoinedCharacterSegmenter(
                width_multiplier=1.8,
                min_character_width=5,
                geometry_split_ratio=geometry_split_ratio,
            ),
        )

    @property
    def ready(self):
        components = (self.model, self.pipeline, self.segmentation, self.segmentation_hypotheses)
        return all(component is not None for component in components)

    def segment(self, image):
        if not self.ready:
            raise RuntimeError(f'Writing model unavailable: {self.error or "unknown error"}')
        return self.segmentation.segment(image.convert('L'), self.model, self.device)

    def recognize(self, image):
        return self._recognize(image, word_only=False)

    def recognize_word(self, image):
        return self._recognize(image, word_only=True)

    def _recognize(self, image, word_only):
        if not self.ready:
            raise RuntimeError(f'Writing model unavailable: {self.error or "unknown error"}')
        started = time.perf_counter()
        segmentation_started = time.perf_counter()
        grayscale = image.convert('L')
        if word_only:
            words = self.segmentation.segment_word_hypotheses(
                grayscale, self.segmentation_hypotheses, self.model, self.device,
            )
            hypotheses = [[[word]] for word in words]
        else:
            hypotheses = self.segmentation.segment_hypotheses(
                grayscale, self.segmentation_hypotheses, self.model, self.device,
            )
        segmentation_ms = (time.perf_counter() - segmentation_started) * 1000

        final_lines, selected = self.pipeline.predict_best_segmentation(
            hypotheses, self.model, self.device,
            method='wordfreq_hybrid', character_top_k=8, beam_width=100,
            frequency_weight=0.50, lm_weight=0.5,
        )
        raw_lines = self.pipeline.predict(selected, self.model, self.device, method='cnn')
        words = sum(len(line) for line in final_lines)
        characters = sum(len(word) for line in selected for word in line)

        return RecognitionResult(
            text='\n'.join([' '.join(word) for word in final_lines]),
            raw_text='\n'.join([' '.join(word) for word in raw_lines]),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            lines=len(selected),
            words=words,
            characters=characters,
            segmentation_ms=segmentation_ms,
        )
