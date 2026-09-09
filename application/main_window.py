from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QThreadPool, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabBar,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .calculator import Calculator
from .canvas import InkCanvas
from .math_recognition import MathRecognizer
from .notes import NoteStore
from .settings import ShortcutSettingsDialog, load_shortcut
from .theme import APP_NAME, AUTOSAVE_MS, STYLESHEET
from .writing_recognition import HandwritingRecognizer, RecognitionWorker


class MarkdownViewer(QTextBrowser):
    edit_intent = Signal(object)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_intent.emit(event.position())
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        editing_keys = (
            Qt.Key.Key_Backspace,
            Qt.Key.Key_Delete,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Tab,
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
        )
        editable = bool(event.text()) or event.key() in editing_keys
        if editable and not event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.edit_intent.emit(event)
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1540, 940)
        self.setMinimumSize(1180, 700)
        self.setStyleSheet(STYLESHEET)

        self.store = NoteStore('notes')
        self.writing = HandwritingRecognizer()
        self.math = MathRecognizer()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)

        self.current_note = None
        self.dirty = False
        self.mode = 'Writing'
        self.math_expression = ''
        self.editing = False
        self._tab_sync = False
        self._recognition_running = False
        self._recognition_pending = False
        self.handwriting_window = None
        self.handwriting_canvas = None
        self.handwriting_info = None
        self.handwriting_expression = None

        self._build_actions()
        self._build_ui()
        self._load_notes()
        self._new_note_if_empty()
        self._start_autosave()

    def _build_actions(self):
        self.actions = {}
        definitions = [
            ('new', 'New note', self.new_note),
            ('open', 'Open note', self.open_external),
            ('save', 'Save', self.save_note),
            ('close', 'Close note', self.close_current_note),
            ('search', 'Search notes', self.focus_search),
            ('handwriting', 'Handwriting input', self.show_handwriting),
            ('calculate', 'Calculate', self.calculate_selected),
            ('edit', 'Edit note', self.toggle_edit_mode),
            ('settings', 'Settings', self.show_settings),
        ]
        for key, text, callback in definitions:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(load_shortcut(key)))
            action.triggered.connect(callback)
            self.actions[key] = action
            self.addAction(action)

        self.editor_shortcuts = []
        editor_shortcuts = (
            ('Ctrl+Backspace', lambda: self._delete_word(True)),
            ('Ctrl+Delete', lambda: self._delete_word(False)),
            ('Ctrl+Shift+Backspace', lambda: self._delete_line(True)),
            ('Ctrl+Shift+Delete', lambda: self._delete_line(False)),
            ('Meta+Backspace', lambda: self._delete_line(True)),
            ('Meta+Delete', lambda: self._delete_line(False)),
        )
        for sequence, callback in editor_shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.editor_shortcuts.append(action)

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        for key in ('new', 'open', 'save'):
            toolbar.addAction(self.actions[key])
        toolbar.addSeparator()
        for text, callback in (
            ('Handwrite', self.show_handwriting),
            ('Calculate', self.calculate_selected),
            ('Search', self.focus_search),
            ('Settings', self.show_settings),
        ):
            toolbar.addWidget(self._toolbar_button(text, callback))
        toolbar.addSeparator()

        self.mode_button = self._toolbar_button('Writing', self._toggle_mode)
        self.mode_button.setObjectName('modeButton')
        self.mode_button.setFixedWidth(94)
        toolbar.addWidget(self.mode_button)

        self.file_tabs = QTabBar()
        self.file_tabs.setExpanding(False)
        self.file_tabs.setMovable(True)
        self.file_tabs.setTabsClosable(False)
        self.file_tabs.setUsesScrollButtons(True)
        self.file_tabs.setDocumentMode(True)
        self.file_tabs.setDrawBase(False)
        self.file_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.file_tabs.currentChanged.connect(self._file_tab_changed)
        toolbar.addWidget(self.file_tabs)
        self.addToolBar(toolbar)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_left_panel())
        root_layout.addWidget(self._build_editor_panel(), 1)
        root_layout.addWidget(self._build_right_panel())
        self.setCentralWidget(root)

        status = QStatusBar()
        self.status_label = QLabel('Ready')
        self.model_label = QLabel(self._model_status())
        self.word_count_label = QLabel('0 words')
        status.addWidget(self.status_label)
        status.addPermanentWidget(self.model_label)
        status.addPermanentWidget(self.word_count_label)
        self.setStatusBar(status)

    def _toolbar_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _build_left_panel(self):
        panel = QFrame()
        panel.setObjectName('sidePanel')
        panel.setFixedWidth(245)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(9, 10, 9, 10)

        title = QLabel('INKNOTE')
        title.setObjectName('appTitle')
        layout.addWidget(title)

        self.search = QLineEdit()
        self.search.setPlaceholderText('Search notes…')
        self.search.textChanged.connect(self._filter_notes)
        layout.addWidget(self.search)
        layout.addWidget(self._section('VAULT'))

        self.note_list = QListWidget()
        self.note_list.currentItemChanged.connect(self._note_selected)
        layout.addWidget(self.note_list, 1)

        new_button = QPushButton('＋  New note')
        new_button.setObjectName('primary')
        new_button.clicked.connect(self.new_note)
        layout.addWidget(new_button)
        return panel

    def _section(self, text):
        label = QLabel(text)
        label.setObjectName('sectionTitle')
        return label

    def _build_editor_panel(self):
        panel = QFrame()
        panel.setObjectName('editorPanel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view_stack = QWidget()
        stack = QVBoxLayout(self.view_stack)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)

        self.viewer = MarkdownViewer()
        self.viewer.setOpenLinks(False)
        self.viewer.setObjectName('viewer')
        self.viewer.setReadOnly(True)
        self.viewer.edit_intent.connect(self._viewer_edit_intent)
        stack.addWidget(self.viewer)

        self.editor = QTextEdit()
        self.editor.setObjectName('editor')
        self.editor.setPlaceholderText('Start writing…')
        self.editor.textChanged.connect(self._editor_changed)
        self.editor.hide()
        stack.addWidget(self.editor)
        layout.addWidget(self.view_stack)
        for action in self.editor_shortcuts:
            self.editor.addAction(action)

        self.view_hint = QLabel('Rendered Markdown · click or type to edit')
        self.view_hint.setObjectName('viewModeLabel')
        layout.addWidget(self.view_hint)
        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setObjectName('rightPanel')
        panel.setFixedWidth(315)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)

        layout.addWidget(self._section('MODE'))
        self.mode_hint = QLabel('Writing mode: words, sentences and notes')
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setObjectName('status')
        layout.addWidget(self.mode_hint)

        layout.addWidget(self._section('TOOLS'))
        for text, callback in (
            ('✎  Handwriting input', self.show_handwriting),
            ('∑  Calculate selection', self.calculate_selected),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            layout.addWidget(button)

        self.math_result = QLabel('Math results will appear here.')
        self.math_result.setWordWrap(True)
        self.math_result.setObjectName('status')
        layout.addWidget(self.math_result)

        layout.addWidget(self._section('DOCUMENT'))
        self.outline = QListWidget()
        self.outline.itemClicked.connect(self._jump_outline)
        layout.addWidget(self.outline, 1)
        return panel

    def _load_notes(self):
        self.note_list.clear()
        self.file_tabs.blockSignals(True)
        while self.file_tabs.count():
            self.file_tabs.removeTab(0)
        for path in self.store.files():
            item = QListWidgetItem(path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.note_list.addItem(item)
            index = self.file_tabs.addTab(path.stem)
            self.file_tabs.setTabData(index, str(path))
        self.file_tabs.blockSignals(False)

    def _new_note_if_empty(self):
        files = self.store.files()
        if files:
            self._select_path(files[0])
        else:
            self.new_note()

    def _select_path(self, path):
        path = Path(path)
        for i in range(self.note_list.count()):
            item_path = Path(self.note_list.item(i).data(Qt.ItemDataRole.UserRole))
            if item_path == path:
                self.note_list.setCurrentRow(i)
                return

    def _note_selected(self, current, previous):
        if current:
            self._load_note(Path(current.data(Qt.ItemDataRole.UserRole)))

    def _file_tab_changed(self, index):
        if self._tab_sync or index < 0:
            return
        path = self.file_tabs.tabData(index)
        if path:
            self._select_path(Path(path))

    def _sync_tabs(self, path):
        for i in range(self.file_tabs.count()):
            if self.file_tabs.tabText(i) == Path(path).stem:
                self._tab_sync = True
                self.file_tabs.setCurrentIndex(i)
                self._tab_sync = False
                return

    def _load_note(self, path):
        if self.dirty and self.current_note:
            self.save_note()
        self.current_note = Path(path)
        text = self.current_note.read_text(encoding='utf-8')
        self.editor.blockSignals(True)
        self.editor.setPlainText(text)
        self.editor.blockSignals(False)
        self._render_note(text)
        self._exit_edit_mode(False)
        self.setWindowTitle(f'{self.current_note.stem} — {APP_NAME}')
        self.dirty = False
        self._update_outline()
        self._update_counts()
        self._sync_tabs(path)
        self.status_label.setText(f'Opened {self.current_note.name}')

    def _render_note(self, text):
        self.viewer.setMarkdown(text or '')

    def new_note(self):
        title, ok = QInputDialog.getText(self, 'New note', 'Note name:')
        if not ok:
            return
        path = self.store.create(title or 'Untitled')
        self._load_notes()
        self._select_path(path)
        self.enter_edit_mode()

    def open_external(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open note', str(self.store.root), 'Markdown (*.md);;Text (*.txt)')
        if path:
            external = Path(path)
            known_paths = {
                Path(self.note_list.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.note_list.count())
            }
            if external not in known_paths:
                item = QListWidgetItem(external.stem)
                item.setData(Qt.ItemDataRole.UserRole, str(external))
                self.note_list.addItem(item)
                index = self.file_tabs.addTab(external.stem)
                self.file_tabs.setTabData(index, str(external))
            self._select_path(external)

    def save_note(self):
        if not self.current_note:
            return
        text = self.editor.toPlainText()
        self.current_note.write_text(text, encoding='utf-8')
        self.dirty = False
        self._render_note(text)
        self.status_label.setText(f'Saved {self.current_note.name}')
        self._load_notes()
        self._select_path(self.current_note)

    def close_current_note(self):
        if self.dirty:
            self.save_note()
        self.close()

    def _editor_changed(self):
        self.dirty = True
        self._render_note(self.editor.toPlainText())
        self._update_outline()
        self._update_counts()

    def _start_autosave(self):
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

    def _autosave(self):
        if not self.dirty or not self.current_note:
            return
        text = self.editor.toPlainText()
        self.current_note.write_text(text, encoding='utf-8')
        self.dirty = False
        self.status_label.setText('Autosaved')
        self._render_note(text)

    def _filter_notes(self, text):
        query = text.casefold().strip()
        for i in range(self.note_list.count()):
            item = self.note_list.item(i)
            item.setHidden(bool(query) and query not in item.text().casefold())

    def _update_counts(self):
        words = len(re.findall(r'\b\w+\b', self.editor.toPlainText()))
        self.word_count_label.setText(f'{words} words')

    def _update_outline(self):
        self.outline.clear()
        for index, line in enumerate(self.editor.toPlainText().splitlines()):
            if re.match(r'^#{1,3}\s+', line):
                item = QListWidgetItem(re.sub(r'^#{1,3}\s+', '', line))
                item.setData(Qt.ItemDataRole.UserRole, index)
                self.outline.addItem(item)

    def _jump_outline(self, item):
        self.enter_edit_mode()
        block_number = item.data(Qt.ItemDataRole.UserRole)
        block = self.editor.document().findBlockByNumber(block_number)
        cursor = self.editor.textCursor()
        cursor.setPosition(block.position())
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()
        self.editor.setFocus()

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    def _toggle_mode(self):
        self.set_mode('Math' if self.mode == 'Writing' else 'Writing')

    def set_mode(self, mode):
        self.mode = mode
        self.mode_button.setText(mode)
        if mode == 'Math':
            self.mode_hint.setText('Math mode: write one symbol at a time → build an expression → result')
            if self.math.ready:
                math_status = 'Math CNN: ready'
            else:
                math_status = f'Math CNN unavailable: {self.math.error or "unknown error"}'
            self.math_result.setText(math_status)
        else:
            self.mode_hint.setText('Writing mode: words, sentences and notes')
            self.math_result.setText('Math results will appear here.')
        self.model_label.setText(self._model_status())
        self._resize_handwriting_window()

    def _model_status(self):
        if self.mode == 'Math':
            return 'Math CNN ready' if self.math.ready else 'Math CNN unavailable'
        return 'Writing CNN ready' if self.writing.ready else 'Writing CNN unavailable'

    def toggle_edit_mode(self, *_):
        if self.editing:
            self._exit_edit_mode(True)
        else:
            self.enter_edit_mode()

    def enter_edit_mode(self, cursor_position=None):
        if not self.current_note:
            return
        if self.editing:
            self.editor.setFocus()
            return
        self.editing = True
        self.viewer.hide()
        self.editor.show()
        self.view_hint.setText('Editing Markdown · Ctrl+E returns to rendered view')
        self.actions['edit'].setText('View note')
        if isinstance(cursor_position, int):
            cursor = self.editor.textCursor()
            cursor.setPosition(min(cursor_position, len(self.editor.toPlainText())))
            self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    def _exit_edit_mode(self, focus=False):
        self.editing = False
        self.editor.hide()
        self.viewer.show()
        self._render_note(self.editor.toPlainText())
        self.view_hint.setText('Rendered Markdown · click or type to edit')
        self.actions['edit'].setText('Edit note')
        if focus:
            self.viewer.setFocus()

    def _viewer_edit_intent(self, payload):
        if isinstance(payload, QKeyEvent):
            self.enter_edit_mode()
            self._replay_key(payload)
            return
        position = self.viewer.cursorForPosition(payload).position() if hasattr(payload, 'x') else None
        self.enter_edit_mode(position)

    def _replay_key(self, event):
        clone = QKeyEvent(
            QEvent.Type.KeyPress,
            event.key(),
            event.modifiers(),
            event.text(),
            event.isAutoRepeat(),
            event.count(),
        )
        QApplication.sendEvent(self.editor, clone)

    def show_settings(self):
        dialog = ShortcutSettingsDialog(self, self.actions)
        if dialog.exec():
            for key, action in self.actions.items():
                action.setShortcut(QKeySequence(load_shortcut(key)))
            self.status_label.setText('Shortcut settings saved')

    def show_handwriting(self):
        if self.handwriting_window is not None:
            self._resize_handwriting_window()
            self.handwriting_window.show()
            self.handwriting_window.raise_()
            self.handwriting_window.activateWindow()
            return

        dialog = QDialog(self, Qt.WindowType.Tool | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
        dialog.setObjectName('handwritingWindow')
        dialog.setStyleSheet(STYLESHEET)
        dialog.setModal(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(6)
        controls = self._build_handwriting_controls(layout)
        canvas, info, expression_label = self._build_handwriting_canvas(layout)
        self._connect_handwriting_controls(controls, dialog, canvas, info, expression_label)
        self._start_recognition_timer(dialog, canvas, info, expression_label)

        dialog.finished.connect(lambda _result: self._handwriting_closed())
        dialog.setLayout(layout)
        self.handwriting_window = dialog
        self.handwriting_canvas = canvas
        self.handwriting_info = info
        self.handwriting_expression = expression_label
        self._resize_handwriting_window()
        dialog.show()
        self._position_handwriting_window()
        dialog.raise_()
        dialog.activateWindow()

    def _build_handwriting_controls(self, layout):
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(QLabel('Pause after writing to convert automatically.'))
        header.addStretch()

        trackpad = QCheckBox('Trackpad draw')
        trackpad.setToolTip(
            'Move the trackpad without holding the mouse button. '
            'The pointer starts near the canvas top-left.'
        )
        clear = QPushButton('Clear')
        recognise = QPushButton('Recognise')
        recognise.setObjectName('primary')
        inspect = QPushButton('Inspect segments')
        insert_ink = QPushButton('Insert ink')
        controls = trackpad, clear, recognise, inspect, insert_ink
        for control in controls:
            header.addWidget(control)
        layout.addLayout(header)
        return controls

    def _build_handwriting_canvas(self, layout):
        expression_label = None
        if self.mode == 'Math':
            expression_label = QLabel('Expression:  —  • draw one symbol, then pause')
            expression_label.setObjectName('mathStream')
            layout.addWidget(expression_label)

            canvas = InkCanvas(pen_width=5, min_size=(190, 190), logical_size=(520, 520))
            canvas.setFixedSize(190, 190)
            canvas_row = QHBoxLayout()
            canvas_row.addStretch()
            canvas_row.addWidget(canvas)
            canvas_row.addStretch()
            layout.addLayout(canvas_row)
            info = QLabel('Draw one symbol in the square.')
            self.math_expression = ''
        else:
            canvas = InkCanvas(pen_width=5, min_size=(560, 180), logical_size=(1400, 520))
            canvas.setFixedSize(760, 170)
            layout.addWidget(canvas, 0, Qt.AlignmentFlag.AlignCenter)
            info = QLabel('')

        info.setObjectName('status')
        layout.addWidget(info)
        return canvas, info, expression_label

    def _connect_handwriting_controls(self, controls, dialog, canvas, info, expression_label):
        trackpad, clear, recognise, inspect, insert_ink = controls
        clear.clicked.connect(canvas.clear)
        trackpad.toggled.connect(lambda enabled: self._set_trackpad_mode(canvas, enabled))
        recognise.clicked.connect(lambda: self._recognise_canvas(dialog, canvas, info, False, expression_label))
        inspect.clicked.connect(lambda: self._inspect_segmentation(dialog, canvas, info))
        insert_ink.clicked.connect(lambda: self._insert_ink(dialog, canvas))

    def _start_recognition_timer(self, dialog, canvas, info, expression_label):
        is_writing = self.mode == 'Writing'
        timer = QTimer(dialog)
        timer.setSingleShot(True)
        timer.setInterval(1100 if is_writing else 1200)
        timer.timeout.connect(lambda: self._recognise_canvas(dialog, canvas, info, True, expression_label))
        canvas.changed.connect(timer.start)
        setattr(self, '_writing_auto_timer' if is_writing else '_math_auto_timer', timer)

    def _position_handwriting_window(self):
        if self.handwriting_window is None:
            return
        screen = self.handwriting_window.screen() or self.screen()
        if screen is None:
            return
        area = screen.availableGeometry()
        margin = 24
        x = area.right() - self.handwriting_window.width() - margin
        y = area.bottom() - self.handwriting_window.height() - margin
        self.handwriting_window.move(max(area.left() + margin, x), max(area.top() + margin, y))

    def _set_trackpad_mode(self, canvas, enabled):
        canvas.set_trackpad_mode(enabled)
        if not enabled:
            return
        canvas.reset_trackpad_pointer()

    def _resize_handwriting_window(self):
        if self.handwriting_window is None:
            return
        self.handwriting_window.setWindowTitle(f'Handwriting — {self.mode} mode')
        size = (330, 320) if self.mode == 'Math' else (820, 260)
        self.handwriting_window.resize(*size)

    def _handwriting_closed(self):
        self._recognition_pending = False
        self.handwriting_window = None
        self.handwriting_canvas = None
        self.handwriting_info = None
        self.handwriting_expression = None
        for timer_name in ('_writing_auto_timer', '_math_auto_timer'):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()

    def _inspect_segmentation(self, dialog, canvas, info):
        if not canvas.has_ink():
            info.setText('Write something first.')
            return
        if not self.writing.ready:
            QMessageBox.warning(
                dialog,
                'Writing CNN unavailable',
                self.writing.error or 'Could not load the writing checkpoint.',
            )
            return
        try:
            started = time.perf_counter()
            segmented = self.writing.segment(canvas.to_pil())
            elapsed = (time.perf_counter() - started) * 1000
            lines = len(segmented)
            words = sum(len(line) for line in segmented)
            chars = sum(len(word) for line in segmented for word in line)
            details = []
            for line_index, line in enumerate(segmented, start=1):
                character_counts = ', '.join(str(len(word)) for word in line)
                details.append(f'Line {line_index}: {len(line)} word(s) — {character_counts} char(s)')
            summary = (
                f'SEGMENTATION ACTIVE  ·  {lines} lines · {words} words '
                f'· {chars} characters · {elapsed:.0f} ms'
            )
            info.setText(f'{summary}\n' + ('\n'.join(details) if details else 'No regions detected.'))
        except Exception as exc:
            info.setText(f'Segmentation error: {exc}')
            QMessageBox.critical(dialog, 'Segmentation error', str(exc))

    def _recognise_canvas(self, dialog, canvas, info, automatic=False, expression_label=None):
        if self._recognition_running:
            self._recognition_pending = True
            return
        if not canvas.has_ink():
            if not automatic:
                QMessageBox.information(dialog, 'Nothing to recognise', 'Write something on the canvas first.')
            return
        if self.mode == 'Math':
            self._recognise_math(dialog, canvas, info, automatic, expression_label)
            return
        if not self.writing.ready:
            if not automatic:
                QMessageBox.warning(
                    dialog,
                    'Writing CNN unavailable',
                    self.writing.error or 'Could not load the writing checkpoint.',
                )
            return
        image = canvas.to_pil()
        info.setText('Segmenting and recognising…')
        self._recognition_running = True
        self._recognition_pending = False
        worker = RecognitionWorker(self.writing, image, dialog, canvas, info, automatic)
        worker.signals.finished.connect(self._writing_done)
        worker.signals.error.connect(self._writing_error)
        self.thread_pool.start(worker)

    def _recognise_math(self, dialog, canvas, info, automatic, expression_label):
        if not self.math.ready:
            if not automatic:
                QMessageBox.warning(
                    dialog,
                    'Math CNN unavailable',
                    self.math.error or 'Could not load the math checkpoint.',
                )
            return
        try:
            candidates = self.math.predict_single(canvas.to_pil(), top_k=3)
            if not candidates:
                info.setText('No symbol detected; try again.')
                return

            prediction, confidence = candidates[0]
            token = self.math.replacements.get(prediction, prediction)
            token = token.replace('\\', '').replace('{', '').replace('}', '').replace(' ', '')
            self.math_expression += token
            answer = self.math.evaluate(self.math_expression)

            if expression_label is not None:
                expression_label.setText(f'Expression: {self.math_expression}   →   {answer or "—"}')
            info.setText(f'Top prediction: {prediction} ({confidence:.0%})')
            self.math_result.setText(f'Expression: {self.math_expression}\nResult: {answer or "—"}')
            canvas.clear()
            self._reset_trackpad_after_recognition(canvas)
        except Exception as exc:
            self._reset_trackpad_after_recognition(canvas)
            info.setText(f'Math recognition error: {exc}')
            if not automatic:
                QMessageBox.critical(dialog, 'Math recognition error', str(exc))

    def _writing_done(self, result, dialog, canvas, info, automatic):
        self._recognition_running = False
        if dialog is None or not dialog.isVisible():
            self._recognition_pending = False
            return
        info.setText(
            f'{result.text or "[no text]"}   ·   raw: {result.raw_text or "[no text]"}'
            f'   ·   {result.lines} lines · {result.words} words · {result.characters} chars'
            f'   ·   segmentation {result.segmentation_ms:.0f} ms   ·   total {result.elapsed_ms:.0f} ms'
        )
        if result.text:
            self._insert_text_at_cursor(result.text)
            self.status_label.setText('Handwriting converted and inserted')
        canvas.clear()
        self._reset_trackpad_after_recognition(canvas)
        if self._recognition_pending:
            self._recognition_pending = False
            if dialog is not None and dialog.isVisible():
                QTimer.singleShot(80, lambda: self._recognise_canvas(dialog, canvas, info, True, None))
        elif not automatic:
            dialog.close()

    def _writing_error(self, error, dialog, canvas, info, automatic):
        self._recognition_running = False
        if dialog is None or not dialog.isVisible():
            self._recognition_pending = False
            return
        self._reset_trackpad_after_recognition(canvas)
        if info is not None:
            info.setText(f'Recognition error: {error}')
        if not automatic:
            QMessageBox.critical(dialog, 'Recognition error', error)
        self._recognition_pending = False

    def _reset_trackpad_after_recognition(self, canvas):
        if canvas is not None and canvas.trackpad_mode:
            canvas.reset_trackpad_pointer()

    def _insert_text_at_cursor(self, text):
        self.enter_edit_mode()
        cursor = self.editor.textCursor()
        existing = self.editor.toPlainText()
        prefix = '' if not existing or existing.endswith(('\n', ' ')) else '\n'
        cursor.insertText(prefix + text)
        self.editor.setTextCursor(cursor)
        self.dirty = True

    def _insert_ink(self, dialog, canvas):
        if not canvas.has_ink() or not self.current_note:
            return
        path = self.store.handwriting / f'ink-{int(time.time() * 1000)}.png'
        canvas.export_png(path)
        relative_path = path.relative_to(self.store.root).as_posix()
        self._insert_text_at_cursor(f'\n\n![Handwritten note]({relative_path})\n')
        dialog.close()
        self.status_label.setText('Handwriting image inserted')

    def calculate_selected(self):
        self.enter_edit_mode()
        cursor = self.editor.textCursor()
        expression = cursor.selectedText().strip() or Calculator.extract(cursor.block().text())
        if not expression:
            QMessageBox.information(self, 'Calculate', 'Select a numeric equation, or place the cursor on one.')
            return
        try:
            result = Calculator.evaluate(expression)
        except Exception as exc:
            QMessageBox.warning(self, 'Calculate', f'Could not calculate:\n{exc}')
            return
        cursor.insertText(f'{expression} = {result}')
        self.dirty = True
        self.status_label.setText(f'Calculated {expression} = {result}')

    def _delete_word(self, backward=True):
        direction = QTextCursor.MoveOperation.PreviousWord if backward else QTextCursor.MoveOperation.NextWord
        self._delete_toward(direction)

    def _delete_line(self, backward=True):
        direction = QTextCursor.MoveOperation.StartOfBlock if backward else QTextCursor.MoveOperation.EndOfBlock
        self._delete_toward(direction)

    def _delete_toward(self, direction):
        self.enter_edit_mode()
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            cursor.movePosition(direction, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self.editor.setTextCursor(cursor)

    def closeEvent(self, event):
        self._recognition_pending = False
        self._recognition_running = False
        for timer_name in ('_writing_auto_timer', '_math_auto_timer'):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        if self.dirty:
            self.save_note()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName('InkNote')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
