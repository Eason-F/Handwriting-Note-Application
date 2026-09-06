import argparse
import io
import sys
from pathlib import Path

import torch
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .data import EMNIST_BYCLASS_CHARACTERS
from .model import WritingCNN
from .prediction import PredictionPipeline
from .segmentation import CharacterSegmenter, ConjoinedCharacterSegmenter, LineSegmenter, SegmentationPipeline, WordSegmenter
from .train import choose_device


class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(700, 700)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.last_point = None
        self.clear()

    def clear(self):
        self.image.fill(QColor("white"))
        self.last_point = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(QColor("#dddddd"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.last_point is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return

        point = event.position().toPoint()
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("black"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.last_point, point)

        self.last_point = point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None

    def as_pillow_image(self):
        encoded = QByteArray()
        buffer = QBuffer(encoded)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.image.save(buffer, "PNG")
        buffer.close()
        return Image.open(io.BytesIO(bytes(encoded))).copy()


class TestWindow(QMainWindow):
    def __init__(self, checkpoint_path: Path, device: torch.device):
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        self.model = WritingCNN(checkpoint["number_of_classes"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device)
        self.model.eval()

        self.device = device

        self.segmentation_pipeline = SegmentationPipeline(
            LineSegmenter(min_ink_pixels=8, max_internal_gap=2, min_line_height=10),
            WordSegmenter(),
            CharacterSegmenter(),
            ConjoinedCharacterSegmenter(width_multiplier=1.8, min_character_width=5),
        )

        self.prediction_pipeline = PredictionPipeline(
            list(EMNIST_BYCLASS_CHARACTERS)
        )

        self.setWindowTitle("Writing Tester")
        self.resize(1200, 800)

        self.canvas = DrawingCanvas()

        self.prediction = QLabel("Draw handwriting, then press Predict")
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.prediction.setMinimumWidth(400)
        self.prediction.setWordWrap(True)
        self.prediction.setStyleSheet("font-size: 24px; padding: 20px;")

        predict_button = QPushButton("Predict")
        clear_button = QPushButton("Clear")

        predict_button.setMinimumHeight(50)
        clear_button.setMinimumHeight(50)
        predict_button.clicked.connect(self.predict)
        clear_button.clicked.connect(self.clear)

        controls = QHBoxLayout()
        controls.addWidget(predict_button)
        controls.addWidget(clear_button)

        left = QVBoxLayout()
        left.addWidget(self.canvas)
        left.addLayout(controls)

        layout = QHBoxLayout()
        layout.addLayout(left)
        layout.addWidget(self.prediction)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def clear(self):
        self.canvas.clear()
        self.prediction.setText("Draw handwriting, then press Predict")

    def predict(self):
        image = self.canvas.as_pillow_image()

        try:
            segmented = self.segmentation_pipeline.segment(
                image,
                self.model,
                self.device,
            )

            raw_lines = []
            beam_lines = []
            reranked_lines = []
            vocabulary_lines = []

            for line in segmented:
                raw_words = []
                beam_words = []
                reranked_words = []
                vocabulary_words = []

                for word in line:
                    raw = ""
                    for character in word:
                        prediction, _ = self.prediction_pipeline.predict_character(
                            self.model,
                            character,
                            self.device,
                        )
                        raw += self.prediction_pipeline.characters[prediction]

                    raw_words.append(raw)

                    beam = self.prediction_pipeline.predict_word_candidates(
                        self.model,
                        word,
                        self.device,
                        k=10,
                        beam_width=100,
                    )

                    if beam:
                        beam_words.append(beam[0][0])

                    reranked = self.prediction_pipeline.rerank_word_candidates(
                        beam,
                        frequency_weight=0.10,
                        language_weight=0.10,
                    )

                    if reranked:
                        reranked_words.append(reranked[0][0])

                    vocabulary = self.prediction_pipeline.predict_word_vocabulary(
                        self.model,
                        word,
                        self.device,
                        k=10,
                        vocabulary_size=50000,
                        frequency_weight=0.10,
                    )

                    if vocabulary:
                        vocabulary_words.append(vocabulary[0][0])

                raw_lines.append(" ".join(raw_words))
                beam_lines.append(" ".join(beam_words))
                reranked_lines.append(" ".join(reranked_words))
                vocabulary_lines.append(" ".join(vocabulary_words))

            raw_text = "\n".join(raw_lines)
            beam_text = "\n".join(beam_lines)
            reranked_text = "\n".join(reranked_lines)
            vocabulary_text = "\n".join(vocabulary_lines)

            if not raw_text:
                self.prediction.setText("No handwriting detected.")
                return

            self.prediction.setText(
                f"Raw CNN:\n{raw_text}\n\n"
                f"Beam:\n{beam_text}\n\n"
                f"Beam + LM:\n{reranked_text}\n\n"
                f"Vocabulary:\n{vocabulary_text}"
            )

        except ValueError as error:
            QMessageBox.information(
                self,
                "Prediction failed",
                str(error),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Segment and recognise handwritten text"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/writing_cnn.pt"),
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train it first with: python -m writing_cnn.train"
        )

    application = QApplication(sys.argv)

    window = TestWindow(
        args.checkpoint,
        choose_device(args.device),
    )

    window.show()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
