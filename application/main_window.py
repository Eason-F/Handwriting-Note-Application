from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QThreadPool, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QStatusBar, QTabBar, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

from .calculator import Calculator
from .canvas import InkCanvas
from .math_recognition import MathRecognizer
from .notes import NoteStore
from .writing_recognition import HandwritingRecognizer, RecognitionWorker
from .theme import APP_NAME, AUTOSAVE_MS, MATH_CHECKPOINT_PATH, NOTES_DIR, STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1540, 940)
        self.setMinimumSize(1180, 700)
        self.setStyleSheet(STYLESHEET)

        self.store = NoteStore(NOTES_DIR)
        self.writing = HandwritingRecognizer()
        self.math = MathRecognizer()
        self.thread_pool = QThreadPool.globalInstance()
        self.current_note = None
        self.dirty = False
        self.mode = 'Writing'
        self.math_expression = ''

        self.open_notes = []

        self._build_actions()
        self._build_ui()
        self._load_notes()
        self._new_note_if_empty()
        self._start_autosave()

    def _build_actions(self):
        self.actions = {}
        definitions = [
            ('new', 'New note', 'Ctrl+N', self.new_note),
            ('open', 'Open note', 'Ctrl+O', self.open_external),
            ('save', 'Save', 'Ctrl+S', self.save_note),
            ('close', 'Close note', 'Ctrl+W', self.close_current_note),
            ('search', 'Search notes', 'Ctrl+F', self.focus_search),
            ('handwriting', 'Handwriting input', 'Ctrl+Shift+H', self.show_handwriting),
            ('calculate', 'Calculate', 'Ctrl+=', self.calculate_selected),
        ]
        for key, text, shortcut, callback in definitions:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.actions[key] = action
            self.addAction(action)

        for index in range(9):
            action = QAction(self)
            action.setShortcut(QKeySequence(f'Ctrl+{index + 1}'))
            action.triggered.connect(lambda checked=False, i=index: self._activate_file_tab(i))
            self.addAction(action)
            meta_action = QAction(self)
            meta_action.setShortcut(QKeySequence(f'Meta+{index + 1}'))
            meta_action.triggered.connect(lambda checked=False, i=index: self._activate_file_tab(i))
            self.addAction(meta_action)

        self.editor_shortcuts = []
        for sequence, callback in (
            ('Ctrl+Backspace', lambda: self._delete_word(backward=True)),
            ('Ctrl+Delete', lambda: self._delete_word(backward=False)),
            ('Ctrl+Shift+Backspace', lambda: self._delete_line(backward=True)),
            ('Ctrl+Shift+Delete', lambda: self._delete_line(backward=False)),
            ('Meta+Backspace', lambda: self._delete_line(backward=True)),
            ('Meta+Delete', lambda: self._delete_line(backward=False)),
        ):
            action = QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.editor_shortcuts.append(action)
            self.editor.addAction(action) if hasattr(self, 'editor') else None

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        for key in ('new', 'open', 'save'):
            toolbar.addAction(self.actions[key])
        toolbar.addSeparator()
        toolbar.addWidget(self._toolbar_button('Handwrite', self.show_handwriting))
        toolbar.addWidget(self._toolbar_button('Calculate', self.calculate_selected))
        toolbar.addWidget(self._toolbar_button('Search', self.focus_search))
        toolbar.addSeparator()
        self.mode_toggle = QPushButton('Writing')
        self.mode_toggle.setObjectName('modeToggle')
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(False)
        self.mode_toggle.setFixedWidth(94)
        self.mode_toggle.setToolTip('Switch between Writing and Math recognition')
        self.mode_toggle.clicked.connect(self._toggle_mode)
        toolbar.addWidget(self.mode_toggle)

        self.file_tabs = QTabBar()
        self.file_tabs.setExpanding(False)
        self.file_tabs.setMovable(True)
        self.file_tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.file_tabs.setUsesScrollButtons(True)
        self.file_tabs.setDocumentMode(True)
        self.file_tabs.setDrawBase(False)
        self.file_tabs.setMinimumWidth(260)
        self.file_tabs.currentChanged.connect(self._file_tab_changed)
        toolbar.addWidget(self.file_tabs)
        self.addToolBar(toolbar)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)
        root_layout.addWidget(self._build_left_panel())
        root_layout.addWidget(self._build_editor_panel(), 1)
        root_layout.addWidget(self._build_right_panel())

        status = QStatusBar()
        self.status_label = QLabel('Ready')
        status.addWidget(self.status_label)
        self.model_label = QLabel(self._model_status())
        status.addPermanentWidget(self.model_label)
        self.word_count_label = QLabel('0 words')
        status.addPermanentWidget(self.word_count_label)
        self.setStatusBar(status)

    def _model_status(self):
        if self.mode == 'Math':
            return 'Math CNN ready' if self.math.ready else 'Math CNN unavailable'
        return 'Writing CNN ready' if self.writing.ready else 'Writing CNN unavailable'

    def _toolbar_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _build_left_panel(self):
        panel = QFrame(); panel.setObjectName('sidePanel'); panel.setFixedWidth(245)
        layout = QVBoxLayout(panel); layout.setContentsMargins(9, 10, 9, 10)
        title = QLabel('INKNOTE'); title.setObjectName('appTitle'); layout.addWidget(title)
        self.search = QLineEdit(); self.search.setPlaceholderText('Search notes…'); self.search.textChanged.connect(self._filter_notes); layout.addWidget(self.search)
        layout.addWidget(self._section('VAULT'))
        self.note_list = QListWidget(); self.note_list.currentItemChanged.connect(self._note_selected); layout.addWidget(self.note_list, 1)
        new = QPushButton('＋  New note'); new.setObjectName('primary'); new.clicked.connect(self.new_note); layout.addWidget(new)
        return panel

    def _section(self, text):
        label = QLabel(text); label.setObjectName('sectionTitle'); return label

    def _build_editor_panel(self):
        panel = QFrame(); panel.setObjectName('editorPanel')
        layout = QVBoxLayout(panel); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True); self.tabs.tabBar().hide()
        self.editor = QTextEdit(); self.editor.setObjectName('editor'); self.editor.setPlaceholderText('Start writing…'); self.editor.textChanged.connect(self._editor_changed)
        self.tabs.addTab(self.editor, 'Note'); layout.addWidget(self.tabs)
        for action in self.editor_shortcuts:
            self.editor.addAction(action)
        return panel

    def _build_right_panel(self):
        panel = QFrame(); panel.setObjectName('rightPanel'); panel.setFixedWidth(315)
        layout = QVBoxLayout(panel); layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(self._section('MODE'))
        self.mode_hint = QLabel('Writing mode: words, sentences and notes')
        self.mode_hint.setWordWrap(True); self.mode_hint.setObjectName('status'); layout.addWidget(self.mode_hint)
        layout.addWidget(self._section('TOOLS'))
        hand = QPushButton('✎  Handwriting input'); hand.clicked.connect(self.show_handwriting); layout.addWidget(hand)
        calc = QPushButton('∑  Calculate selection'); calc.clicked.connect(self.calculate_selected); layout.addWidget(calc)
        self.math_result = QLabel('Math results will appear here.')
        self.math_result.setWordWrap(True); self.math_result.setObjectName('status'); layout.addWidget(self.math_result)
        layout.addWidget(self._section('DOCUMENT'))
        self.outline = QListWidget(); self.outline.itemClicked.connect(self._jump_outline); layout.addWidget(self.outline, 1)
        return panel

    def _load_notes(self):
        self.note_list.clear()
        for path in self.store.files():
            item = QListWidgetItem(path.stem); item.setData(Qt.ItemDataRole.UserRole, str(path)); self.note_list.addItem(item)

    def _new_note_if_empty(self):
        files = self.store.files()
        self._select_path(files[0]) if files else self.new_note()

    def _select_path(self, path):
        for i in range(self.note_list.count()):
            item = self.note_list.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == Path(path): self.note_list.setCurrentItem(item); return

    def _note_selected(self, current, previous):
        if current: self._load_note(Path(current.data(Qt.ItemDataRole.UserRole)))

    def _load_note(self, path):
        if self.dirty and self.current_note: self.save_note()
        self.current_note = Path(path)
        self.editor.blockSignals(True); self.editor.setPlainText(self.current_note.read_text(encoding='utf-8')); self.editor.blockSignals(False)
        self.setWindowTitle(f'{self.current_note.stem} — {APP_NAME}')
        self._ensure_file_tab(self.current_note)
        self.dirty = False; self._update_outline(); self._update_counts(); self.status_label.setText(f'Opened {self.current_note.name}')

    def _activate_file_tab(self, index):
        if 0 <= index < self.file_tabs.count():
            self.file_tabs.setCurrentIndex(index)

    def _ensure_file_tab(self, path):
        path = Path(path)
        for index in range(self.file_tabs.count()):
            if Path(self.file_tabs.tabData(index)) == path:
                self.file_tabs.blockSignals(True); self.file_tabs.setCurrentIndex(index); self.file_tabs.blockSignals(False)
                return index
        index = self.file_tabs.addTab(path.stem)
        self.file_tabs.setTabData(index, str(path))
        self.file_tabs.setTabToolTip(index, str(path))
        self.file_tabs.setTabText(index, path.stem)
        self.file_tabs.setCurrentIndex(index)
        return index

    def _file_tab_changed(self, index):
        if index < 0: return
        path = self.file_tabs.tabData(index)
        if path and Path(path) != self.current_note:
            self._load_note(Path(path))

    def new_note(self):
        title, ok = QInputDialog.getText(self, 'New note', 'Note name:')
        if not ok: return
        path = self.store.create(title or 'Untitled'); self._load_notes(); self._select_path(path)

    def open_external(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open note', str(self.store.root), 'Markdown (*.md);;Text (*.txt)')
        if path: self._load_note(Path(path))

    def save_note(self):
        if not self.current_note: return
        self.current_note.write_text(self.editor.toPlainText(), encoding='utf-8'); self.dirty = False; self.status_label.setText(f'Saved {self.current_note.name}')
        self._load_notes(); self._select_path(self.current_note)

    def close_current_note(self):
        if self.dirty: self.save_note()
        self.close()

    def _editor_changed(self):
        self.dirty = True; self._update_outline(); self._update_counts()

    def _start_autosave(self):
        self.autosave_timer = QTimer(self); self.autosave_timer.setInterval(AUTOSAVE_MS); self.autosave_timer.timeout.connect(self._autosave); self.autosave_timer.start()

    def _autosave(self):
        if self.dirty and self.current_note:
            self.current_note.write_text(self.editor.toPlainText(), encoding='utf-8'); self.dirty = False; self.status_label.setText('Autosaved')

    def _filter_notes(self, text):
        query = text.casefold().strip()
        for i in range(self.note_list.count()):
            item = self.note_list.item(i); item.setHidden(bool(query) and query not in item.text().casefold())

    def _update_counts(self):
        words = len(re.findall(r'\b\w+\b', self.editor.toPlainText())); self.word_count_label.setText(f'{words} words')

    def _update_outline(self):
        self.outline.clear()
        for index, line in enumerate(self.editor.toPlainText().splitlines()):
            if re.match(r'^#{1,3}\s+', line):
                item = QListWidgetItem(re.sub(r'^#{1,3}\s+', '', line)); item.setData(Qt.ItemDataRole.UserRole, index); self.outline.addItem(item)

    def _jump_outline(self, item):
        block = self.editor.document().findBlockByNumber(item.data(Qt.ItemDataRole.UserRole)); cursor = self.editor.textCursor(); cursor.setPosition(block.position()); self.editor.setTextCursor(cursor); self.editor.ensureCursorVisible(); self.editor.setFocus()

    def focus_search(self):
        self.search.setFocus(); self.search.selectAll()

    def _toggle_mode(self, checked):
        self.set_mode('Math' if checked else 'Writing')

    def set_mode(self, mode):
        self.mode = mode
        if hasattr(self, 'mode_toggle'):
            self.mode_toggle.blockSignals(True)
            self.mode_toggle.setChecked(mode == 'Math')
            self.mode_toggle.setText('Math' if mode == 'Math' else 'Writing')
            self.mode_toggle.blockSignals(False)
        if mode == 'Math':
            self.mode_hint.setText('Math mode: handwritten digits and symbols → equation → result')
            self.math_result.setText('Math CNN: ready' if self.math.ready else f'Math CNN unavailable: {self.math.error or "unknown error"}')
        else:
            self.mode_hint.setText('Writing mode: words, sentences and notes')
            self.math_result.setText('Math results will appear here.')
        self.model_label.setText(self._model_status())

    def show_handwriting(self):
        dialog = QWidget(self, Qt.WindowType.Window)
        dialog.setWindowTitle(f'Handwriting — {self.mode} mode')
        dialog.resize(1000, 650); dialog.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout(dialog)
        header = QHBoxLayout(); header.addWidget(QLabel('Write naturally. Recognition uses the active mode.'))
        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); header.addWidget(spacer)
        clear = QPushButton('Clear'); header.addWidget(clear)
        recognise = QPushButton('Recognise'); recognise.setObjectName('primary'); header.addWidget(recognise)
        insert_ink = QPushButton('Insert ink'); header.addWidget(insert_ink); layout.addLayout(header)
        canvas = InkCanvas(pen_width=3 if self.mode == 'Writing' else 4); layout.addWidget(canvas, 1)
        info = QLabel(''); info.setObjectName('status'); layout.addWidget(info)
        clear.clicked.connect(canvas.clear)
        recognise.clicked.connect(lambda: self._recognise_canvas(dialog, canvas, info))
        insert_ink.clicked.connect(lambda: self._insert_ink(dialog, canvas))
        dialog.show(); dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose); self.handwriting_window = dialog

    def _recognise_canvas(self, dialog, canvas, info):
        if not canvas.has_ink(): QMessageBox.information(dialog, 'Nothing to recognise', 'Write something on the canvas first.'); return
        if self.mode == 'Math':
            if not self.math.ready: QMessageBox.warning(dialog, 'Math CNN unavailable', self.math.error or 'Could not load the math checkpoint.'); return
            try:
                result = self.math.recognize(canvas.to_pil())
                info.setText(f'{result.expression}    =    {result.answer}   ·   {result.elapsed_ms:.0f} ms')
                self.math_result.setText(f'Expression: {result.expression}\nResult: {result.answer}\n{result.symbols} symbols · {result.elapsed_ms:.0f} ms')
                self._insert_text_at_cursor(result.expression)
                if result.answer and result.answer != '—': self._insert_text_at_cursor(f' = {result.answer}')
                dialog.close()
            except Exception as exc:
                QMessageBox.critical(dialog, 'Math recognition error', str(exc))
            return
        if not self.writing.ready: QMessageBox.warning(dialog, 'Writing CNN unavailable', self.writing.error or 'Could not load the writing checkpoint.'); return
        info.setText('Recognising…')
        worker = RecognitionWorker(self.writing, canvas.to_pil())
        worker.signals.finished.connect(lambda result: self._writing_done(dialog, result, info))
        worker.signals.error.connect(lambda error: QMessageBox.critical(dialog, 'Recognition error', error))
        self.thread_pool.start(worker)

    def _writing_done(self, dialog, result, info):
        info.setText(f'{result.text or "[no text]"}   ·   {result.elapsed_ms:.0f} ms')
        if result.text: self._insert_text_at_cursor(result.text); self.status_label.setText('Handwriting converted and inserted'); dialog.close()

    def _insert_text_at_cursor(self, text):
        cursor = self.editor.textCursor(); existing = self.editor.toPlainText()
        prefix = '' if not existing or existing.endswith(('\n', ' ')) else '\n'
        cursor.insertText(prefix + text); self.editor.setTextCursor(cursor); self.dirty = True

    def _insert_ink(self, dialog, canvas):
        if not canvas.has_ink() or not self.current_note: return
        path = self.store.handwriting / f'ink-{int(time.time() * 1000)}.png'; canvas.export_png(path)
        relative = path.relative_to(self.store.root); self._insert_text_at_cursor(f'\n\n![Handwritten note]({relative.as_posix()})\n'); dialog.close(); self.status_label.setText('Handwriting image inserted')

    def calculate_selected(self):
        expression = self.editor.textCursor().selectedText().strip() or Calculator.extract(self.editor.textCursor().block().text())
        if not expression: QMessageBox.information(self, 'Calculate', 'Select a numeric equation, or place the cursor on one.'); return
        try:
            result = Calculator.evaluate(expression)
        except Exception as exc:
            QMessageBox.warning(self, 'Calculate', f'Could not calculate:\n{exc}'); return
        cursor = self.editor.textCursor(); cursor.insertText(f'{expression} = {result}'); self.dirty = True; self.status_label.setText(f'Calculated {expression} = {result}')

    def _delete_word(self, backward=True):
        cursor = self.editor.textCursor()
        if cursor.hasSelection(): cursor.removeSelectedText(); return
        if backward:
            cursor.movePosition(QTextCursor.MoveOperation.PreviousWord, QTextCursor.MoveMode.KeepAnchor)
        else:
            cursor.movePosition(QTextCursor.MoveOperation.NextWord, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText(); self.editor.setTextCursor(cursor)

    def _delete_line(self, backward=True):
        cursor = self.editor.textCursor()
        if cursor.hasSelection(): cursor.removeSelectedText(); return
        if backward:
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
        else:
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText(); self.editor.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.dirty: self.save_note()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle('Fusion')
    window = MainWindow(); window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
