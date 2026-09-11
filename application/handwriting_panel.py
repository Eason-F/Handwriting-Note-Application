from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QTextEdit, QVBoxLayout
from PySide6.QtCore import Qt

from .canvas import InkCanvas


class HandwritingPanel(QFrame):
    recognise_requested = Signal(bool)
    inspect_requested = Signal()
    insert_ink_requested = Signal()
    close_requested = Signal()
    cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('handwritingPanel')
        self.setMinimumHeight(220)
        self.mode = 'text'
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(7)

        controls = QHBoxLayout()
        title = QLabel('HANDWRITING INPUT')
        title.setObjectName('sectionTitle')
        controls.addWidget(title)
        controls.addStretch()
        self.trackpad = QCheckBox('Trackpad draw')
        controls.addWidget(self.trackpad)
        controls.addWidget(QLabel('Size'))
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(2, 14)
        self.width_slider.setValue(5)
        self.width_slider.setFixedWidth(90)
        controls.addWidget(self.width_slider)
        for text, primary in (('Clear', False), ('Inspect', False), ('Recognise & insert', True), ('Keep as ink', False), ('Close', False)):
            button = QPushButton(text)
            if primary:
                button.setObjectName('primary')
            controls.addWidget(button)
            if text == 'Clear':
                button.clicked.connect(self._clear)
            elif text == 'Inspect':
                button.clicked.connect(self.inspect_requested.emit)
            elif text == 'Recognise & insert':
                button.clicked.connect(lambda: self.recognise_requested.emit(False))
            elif text == 'Keep as ink':
                button.clicked.connect(self.insert_ink_requested.emit)
            else:
                button.clicked.connect(self.close_requested.emit)
        layout.addLayout(controls)

        self.canvas = InkCanvas(pen_width=5, min_size=(640, 145), logical_size=(1400, 420))
        layout.addWidget(self.canvas, 1, Qt.AlignmentFlag.AlignHCenter)
        self.preview = QTextEdit()
        self.preview.setPlaceholderText('Recognition preview appears here. You can correct it before inserting.')
        self.preview.setFixedHeight(54)
        layout.addWidget(self.preview)
        self.trackpad.toggled.connect(self._trackpad_changed)
        self.width_slider.valueChanged.connect(self.canvas.set_pen_width)
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(1200)
        self.auto_timer.timeout.connect(lambda: self.recognise_requested.emit(True))
        self.canvas.changed.connect(self.auto_timer.start)

    def _clear(self):
        self.canvas.clear()
        self.preview.clear()
        self.cleared.emit()

    def _trackpad_changed(self, enabled):
        self.canvas.set_trackpad_mode(enabled)
        if enabled:
            self.canvas.reset_trackpad_pointer()

    def set_busy(self, busy):
        self.preview.setReadOnly(busy)
        if busy:
            self.preview.setPlainText('Segmenting and recognising…')

    def configure_mode(self, mode):
        self.mode = mode
        self.preview.clear()
        self.canvas.clear()
        self.auto_timer.stop()
        if mode == 'math':
            self.canvas.set_stretch_to_fill(False)
            self.canvas.set_logical_size((520, 520))
            self.canvas.setMinimumWidth(220)
            self.canvas.setMaximumWidth(300)
            self.preview.setPlaceholderText('Draw one mathematical symbol. Recognition starts after a short pause.')
        else:
            self.canvas.set_logical_size((1400, 700))
            self.canvas.set_stretch_to_fill(True)
            self.canvas.setMinimumWidth(500)
            self.canvas.setMaximumWidth(16777215)
            self.preview.setPlaceholderText('Recognition preview appears here. You can correct it before inserting.')
        self.auto_timer.stop()
