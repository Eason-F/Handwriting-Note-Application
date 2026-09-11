from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def outline_icon(name: str, color='#b7c1ba', size=18):
    """Return a lightweight, platform-independent outline icon."""
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if name == 'file':
        path = QPainterPath(QPointF(5, 2.5))
        path.lineTo(11, 2.5)
        path.lineTo(14.5, 6)
        path.lineTo(14.5, 15.5)
        path.lineTo(5, 15.5)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(11, 2.5), QPointF(11, 6))
        painter.drawLine(QPointF(11, 6), QPointF(14.5, 6))
    elif name == 'folder':
        path = QPainterPath(QPointF(2.5, 5))
        path.lineTo(7.5, 5)
        path.lineTo(9, 7)
        path.lineTo(15.5, 7)
        path.lineTo(14.5, 14.5)
        path.lineTo(3.5, 14.5)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == 'new':
        painter.drawRoundedRect(QRectF(3, 3, 12, 12), 2, 2)
        painter.drawLine(QPointF(9, 6), QPointF(9, 12))
        painter.drawLine(QPointF(6, 9), QPointF(12, 9))
    elif name == 'open':
        painter.drawPath(QPainterPath(QPointF(2.5, 7)).simplified())
        painter.drawLine(QPointF(3, 7), QPointF(15, 7))
        painter.drawLine(QPointF(15, 7), QPointF(13.5, 14.5))
        painter.drawLine(QPointF(13.5, 14.5), QPointF(4, 14.5))
        painter.drawLine(QPointF(4, 14.5), QPointF(2.5, 7))
        painter.drawLine(QPointF(4, 7), QPointF(5, 4.5))
        painter.drawLine(QPointF(5, 4.5), QPointF(9, 4.5))
        painter.drawLine(QPointF(9, 4.5), QPointF(10.5, 7))
    elif name == 'save':
        painter.drawRoundedRect(QRectF(3, 2.5, 12, 13), 2, 2)
        painter.drawRect(QRectF(6, 2.5, 6, 4))
        painter.drawRoundedRect(QRectF(6, 10, 6, 5.5), 1, 1)
    elif name == 'settings':
        painter.drawEllipse(QPointF(9, 9), 2.5, 2.5)
        painter.drawEllipse(QPointF(9, 9), 6, 6)
        for start, end in (((9, 1.5), (9, 3)), ((9, 15), (9, 16.5)), ((1.5, 9), (3, 9)), ((15, 9), (16.5, 9))):
            painter.drawLine(QPointF(*start), QPointF(*end))
    elif name == 'refresh':
        painter.drawArc(QRectF(3, 3, 12, 12), 35 * 16, 285 * 16)
        painter.drawLine(QPointF(13.5, 3.5), QPointF(14.7, 6.5))
        painter.drawLine(QPointF(13.5, 3.5), QPointF(10.5, 4.2))

    painter.end()
    return QIcon(pixmap)
