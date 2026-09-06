from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class InkCanvas(QWidget):
    changed = Signal()

    def __init__(self, background='white', pen_width=3, min_size=(640, 360)):
        super().__init__()
        self.setMinimumSize(*min_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.background = QColor(background)
        self.pen_color = QColor('#111111')
        self.pen_width = pen_width
        self.image = QImage(1400, 820, QImage.Format.Format_ARGB32)
        self.image.fill(self.background)
        self.last_point = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self.image)

    def resizeEvent(self, event):
        if self.width() <= 0 or self.height() <= 0:
            return
        if self.image.width() == self.width() and self.image.height() == self.height():
            return
        new_image = QImage(max(self.width(), 640), max(self.height(), 360), QImage.Format.Format_ARGB32)
        new_image.fill(self.background)
        painter = QPainter(new_image)
        painter.drawImage(QRectF(0, 0, new_image.width(), new_image.height()), self.image)
        painter.end()
        self.image = new_image
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position()

    def mouseMoveEvent(self, event):
        if self.last_point is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        point = event.position()
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.last_point, point)
        painter.end()
        self.last_point = point
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None

    def clear(self):
        self.image.fill(self.background)
        self.update()
        self.changed.emit()

    def to_pil(self):
        image = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        return Image.frombytes('L', (image.width(), image.height()), bytes(image.bits()))

    def has_ink(self):
        return min(self.to_pil().getextrema()) < 245

    def export_png(self, path):
        self.image.save(str(path), 'PNG')
