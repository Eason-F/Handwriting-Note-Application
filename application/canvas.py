import io
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class InkCanvas(QWidget):
    changed = Signal()

    def __init__(self, background='#F6F7F8', pen_width=5, min_size=(640, 360), logical_size=None):
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
        self.current_stroke = None
        self.strokes = []
        self.tool = 'pen'
        self.stretch_to_fill = False
        self.trackpad_mode = False
        self.trackpad_screen_rect = None
        self._ignore_next_trackpad_move = False
        self.revision = 0
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        painter.drawImage(self._canvas_rect(), self.image)

    def resizeEvent(self, event):
        if self.stretch_to_fill and event.size().width() > 0 and event.size().height() > 0:
            self.image = self.image.scaled(event.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logical_size = (self.image.width(), self.image.height())
        super().resizeEvent(event)

    def _canvas_rect(self):
        if self.width() <= 0 or self.height() <= 0:
            return QRectF()
        if self.stretch_to_fill:
            return QRectF(0, 0, self.width(), self.height())
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _image_point(self, point):
        rect = self._canvas_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return QPointF()
        x = min(max(point.x() - rect.left(), 0), rect.width())
        y = min(max(point.y() - rect.top(), 0), rect.height())
        return QPointF(x * self.image.width() / rect.width(), y * self.image.height() / rect.height())

    def _image_pen_width(self):
        rect = self._canvas_rect()
        if rect.width() <= 0:
            return self.pen_width
        return max(1.0, self.pen_width * self.image.width() / rect.width())

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
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.background if self.tool == 'eraser' else self.pen_color
        width = self._image_pen_width() * (3 if self.tool == 'eraser' else 1)
        pen = QPen(
            color,
            width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)

        if self.last_point is None:
            self.last_point = point
            self.current_stroke = {'tool': self.tool, 'width': self.pen_width, 'points': []}
            self.strokes.append(self.current_stroke)
            painter.drawPoint(point)
        else:
            painter.drawLine(self.last_point, point)
            self.last_point = point
        self.current_stroke['points'].append([round(point.x(), 2), round(point.y(), 2)])

        painter.end()
        self.revision += 1
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
            self.current_stroke = None

    def leaveEvent(self, event):
        if self.trackpad_mode:
            self.last_point = None
        super().leaveEvent(event)

    def clear(self):
        self.image.fill(self.background)
        self.strokes = []
        self.revision += 1
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

    def set_tool(self, tool):
        if tool not in {'pen', 'eraser'}:
            raise ValueError(f'Unknown drawing tool: {tool}')
        self.tool = tool
        self.setCursor(Qt.CursorShape.CrossCursor if tool == 'pen' else Qt.CursorShape.PointingHandCursor)

    def set_pen_width(self, width):
        self.pen_width = max(1, int(width))

    def set_stretch_to_fill(self, enabled):
        self.stretch_to_fill = bool(enabled)
        if enabled and self.width() > 0 and self.height() > 0:
            self.image = self.image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logical_size = (self.image.width(), self.image.height())
        self.update()

    def set_logical_size(self, size):
        if tuple(size) == (self.image.width(), self.image.height()):
            return
        self.logical_size = tuple(size)
        self.image = QImage(self.logical_size[0], self.logical_size[1], QImage.Format.Format_ARGB32)
        self.image.fill(self.background)
        self.strokes = []
        self.last_point = None
        self.revision += 1
        self.update()
        self.changed.emit()

    def get_strokes(self):
        return [{'tool': stroke['tool'], 'width': stroke['width'], 'points': [list(point) for point in stroke['points']]} for stroke in self.strokes]

    def set_strokes(self, strokes):
        self.strokes = []
        self.image.fill(self.background)
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for raw in strokes or []:
            points = raw.get('points', [])
            if not points:
                continue
            stroke = {'tool': raw.get('tool', 'pen'), 'width': int(raw.get('width', 5)), 'points': points}
            self.strokes.append(stroke)
            color = self.background if stroke['tool'] == 'eraser' else self.pen_color
            width = stroke['width'] * (3 if stroke['tool'] == 'eraser' else 1)
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            previous = QPointF(*points[0])
            painter.drawPoint(previous)
            for coordinates in points[1:]:
                point = QPointF(*coordinates)
                painter.drawLine(previous, point)
                previous = point
        painter.end()
        self.revision += 1
        self.update()
