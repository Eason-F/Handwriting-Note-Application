import argparse
import io
import sys
from pathlib import Path

import torch
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPoint, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .images import crop_and_center_drawing, prepare_symbol_image
from .model import SymbolCNN
from .train import choose_device


class DrawingCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(360, 360)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.last_point: QPoint | None = None
        self.clear()

    def clear(self) -> None:
        self.image.fill(QColor("white"))
        self.last_point = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(QColor("#dddddd"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.last_point is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        point = event.position().toPoint()
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                QColor("black"),
                18,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawLine(self.last_point, point)
        self.last_point = point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None

    def as_pillow_image(self) -> Image.Image:
        encoded = QByteArray()
        buffer = QBuffer(encoded)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.image.save(buffer, "PNG")
        buffer.close()
        return Image.open(io.BytesIO(bytes(encoded))).copy()


class TestWindow(QMainWindow):
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.symbols = checkpoint["symbols"]
        self.model = SymbolCNN(checkpoint["number_of_classes"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()
        self.device = device

        self.setWindowTitle("HASYv2 Symbol Tester")
        self.canvas = DrawingCanvas()
        self.prediction = QLabel("Draw one symbol, then press Predict")
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.prediction.setMinimumWidth(230)
        self.prediction.setStyleSheet("font-size: 18px; padding: 12px;")

        predict_button = QPushButton("Predict")
        clear_button = QPushButton("Clear")
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

    def clear(self) -> None:
        self.canvas.clear()
        self.prediction.setText("Draw one symbol, then press Predict")

    def predict(self) -> None:
        image = crop_and_center_drawing(self.canvas.as_pillow_image())
        if image is None:
            QMessageBox.information(self, "Nothing drawn", "Draw a symbol first.")
            return

        tensor = prepare_symbol_image(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        values, indexes = probabilities.topk(5)

        lines = ["Top predictions:", ""]
        for rank, (value, index) in enumerate(zip(values.tolist(), indexes.tolist()), 1):
            latex = self.symbols[index]["latex"]
            lines.append(f"{rank}. {latex}   {value:.1%}")
        self.prediction.setText("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Draw and classify a HASYv2 symbol")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/math_cnn.pt"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train it first with: python -m math_cnn.train"
        )

    application = QApplication(sys.argv)
    window = TestWindow(args.checkpoint, choose_device(args.device))
    window.show()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()

