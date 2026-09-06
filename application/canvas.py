import io
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class InkCanvas(QWidget):
    changed = Signal()

    def __init__(self, background='white', pen_width=5, min_size=(640, 360), logical_size=None):
        super().__init__()
        self.setMinimumSize(*min_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.background = QColor(background)
        self.pen_color = QColor('#111111')
        self.pen_width = pen_width
        self.logical_size = logical_size or ((1400, 520) if min_size[0] >= 500 else (520, 520))
        self.image = QImage(self.logical_size[0], self.logical_size[1], QImage.Format.Format_ARGB32)
        self.image.fill(self.background)
        self.last_point = None
        self.trackpad_mode = False
        self.trackpad_screen_rect = None
        self._ignore_next_trackpad_move = False
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self.image)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _image_point(self, point):
        if self.width() <= 0 or self.height() <= 0:
            return QPointF()
        return QPointF(point.x() * self.image.width() / self.width(), point.y() * self.image.height() / self.height())

    def _image_pen_width(self):
        if self.width() <= 0:
            return self.pen_width
        return max(1.0, self.pen_width * self.image.width() / self.width())

    def set_trackpad_mode(self, enabled):
        self.trackpad_mode = bool(enabled)
        self.setMouseTracking(self.trackpad_mode)
        self.last_point = None

    def reset_trackpad_pointer(self):
        self.last_point = None
        self._ignore_next_trackpad_move = True
        point = self.mapToGlobal(QPointF(14, 14).toPoint())
        QCursor.setPos(point)

    def set_trackpad_screen_rect(self, rect):
        self.trackpad_screen_rect = rect

    def _trackpad_point(self, global_position):
        rect = self.trackpad_screen_rect
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return self._image_point(self.mapFromGlobal(global_position))
        x = min(max((global_position.x() - rect.left()) / rect.width(), 0.0), 1.0)
        y = min(max((global_position.y() - rect.top()) / rect.height(), 0.0), 1.0)
        return QPointF(x * self.image.width(), y * self.image.height())

    def _draw_to(self, point):
        if self.last_point is None:
            self.last_point = point
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.pen_color, self._image_pen_width(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPoint(point)
            painter.end()
            self.update()
            self.changed.emit()
            return
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.pen_color, self._image_pen_width(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.last_point, point)
        painter.end()
        self.last_point = point
        self.update()
        self.changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.trackpad_mode:
            self._draw_to(self._image_point(event.position()))

    def mouseMoveEvent(self, event):
        if self.trackpad_mode:
            if self._ignore_next_trackpad_move:
                self._ignore_next_trackpad_move = False
                return
            self._draw_to(self._trackpad_point(self.mapToGlobal(event.position())))
            return
        if self.last_point is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        self._draw_to(self._image_point(event.position()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.trackpad_mode:
            self.last_point = None

    def leaveEvent(self, event):
        if self.trackpad_mode:
            self.last_point = None
        super().leaveEvent(event)

    def clear(self):
        self.image.fill(self.background)
        self.update()
        self.changed.emit()

    def to_pil(self):
        encoded = QByteArray()
        buffer = QBuffer(encoded)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.image.save(buffer, 'PNG')
        buffer.close()
        return Image.open(io.BytesIO(bytes(encoded))).convert('L').copy()

    def has_ink(self):
        return min(self.to_pil().getextrema()) < 245

    def export_png(self, path):
        self.image.save(str(path), 'PNG')
