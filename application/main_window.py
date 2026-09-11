from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QThreadPool, Qt, QTimer
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QTextCursor
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu,
    QMessageBox, QPushButton, QSplitter, QStatusBar, QTextEdit,
    QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .calculator import MathEvaluationError, MathEvaluator
from .handwriting_panel import HandwritingPanel
from .math_recognition import MathRecognizer
from .notes import NoteDocument, NoteError, NoteManager
from .settings import ShortcutSettingsDialog, load_shortcut
from .theme import APP_NAME, AUTOSAVE_MS, STYLESHEET
from .writing_recognition import HandwritingRecognizer, RecognitionWorker


class EditorWidget(QWidget):
    """Owns the main text editing widget without owning document state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text = QTextEdit()
        self.text.setObjectName('editor')
        self.text.setPlaceholderText('Start writing…')
        self.text.setAcceptRichText(False)
        layout.addWidget(self.text)

    def load_document(self, document):
        self.text.blockSignals(True)
        self.text.setPlainText(document.text)
        self.text.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1420, 900)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(STYLESHEET)
        self.manager = NoteManager('notes')
        self.store = self.manager
        self.document: NoteDocument | None = None
        self.current_note: Path | None = None
        self.math_contexts: dict[Path, MathEvaluator] = {}
        self.writing = HandwritingRecognizer()
        self.math_recognizer = MathRecognizer()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)
        self._recognition_running = False
        self._writing_preview = None
        self._writing_preview_revision = None
        self.mode = 'text'
        self.input_mode = 'text'
        self.math_expression = ''

        self._build_actions()
        self._build_ui()
        self._load_tree()
        notes = self.manager.files()
        self._open_path(notes[0]) if notes else self.new_note(default_name='Welcome')
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

    @property
    def dirty(self):
        return bool(self.document and self.document.dirty)

    @dirty.setter
    def dirty(self, value):
        if self.document:
            self.document.dirty = value

    def _build_actions(self):
        self.actions = {}
        definitions = (
            ('new', 'New note', self.new_note), ('open', 'Open…', self.open_external),
            ('save', 'Save', self.save_note), ('save_as', 'Save As…', self.save_as),
            ('close', 'Close', self.close), ('search', 'Search notes', self.focus_search),
            ('handwriting', 'Handwriting panel', self.toggle_handwriting_panel),
            ('calculate', 'Evaluate', lambda: self.evaluate_expression(insert=True)), ('edit', 'Text/Math mode', self.toggle_input_mode),
            ('settings', 'Settings', self.show_settings),
        )
        for key, label, callback in definitions:
            action = QAction(label, self)
            shortcut = load_shortcut(key)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.actions[key] = action
            self.addAction(action)
        for shortcut, callback in (
            ('Ctrl+Shift+S', self.save_as), ('Ctrl+Shift+=', self.insert_math_result),
            ('Ctrl+Z', lambda: self.editor.text.undo()), ('Ctrl+Shift+Z', lambda: self.editor.text.redo()),
        ):
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        for key in ('new', 'open', 'save', 'save_as'):
            toolbar.addAction(self.actions[key])
        toolbar.addSeparator()
        self.input_mode_button = QPushButton('Text')
        self.input_mode_button.setObjectName('modeButton')
        self.input_mode_button.clicked.connect(self.toggle_input_mode)
        toolbar.addWidget(self.input_mode_button)
        self.panel_mode_button = QPushButton('Handwriting input')
        self.panel_mode_button.setCheckable(True)
        self.panel_mode_button.clicked.connect(self.toggle_handwriting_panel)
        toolbar.addWidget(self.panel_mode_button)
        toolbar.addSeparator()
        toolbar.addAction(self.actions['calculate'])
        insert_result = QAction('Insert result', self)
        insert_result.triggered.connect(self.insert_math_result)
        toolbar.addAction(insert_result)
        self.addToolBar(toolbar)

        self.sidebar = QWidget()
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(9, 10, 9, 10)
        heading = QLabel('INKNOTE LIBRARY')
        heading.setObjectName('appTitle')
        side_layout.addWidget(heading)
        self.search = QTextEdit()
        self.search.setPlaceholderText('Filter notes…')
        self.search.setFixedHeight(38)
        self.search.textChanged.connect(self._filter_tree)
        side_layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(lambda item, _column: self._activate_item(item))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        side_layout.addWidget(self.tree, 1)
        buttons = QHBoxLayout()
        for text, callback in (('+ Note', self.new_note), ('+ Folder', self.new_folder), ('↻', self.refresh)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        side_layout.addLayout(buttons)

        self.editor = EditorWidget()
        self.editor.text.textChanged.connect(self._text_changed)
        self.handwriting_panel = HandwritingPanel()
        self.handwriting_panel.hide()
        self.handwriting_panel.close_requested.connect(self.toggle_handwriting_panel)
        self.handwriting_panel.recognise_requested.connect(self._recognise_panel)
        self.handwriting_panel.inspect_requested.connect(self._inspect_panel)
        self.handwriting_panel.insert_ink_requested.connect(self._insert_panel_as_image)
        self.handwriting_panel.cleared.connect(self._clear_math_expression)

        editor_container = QWidget()
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.addWidget(self.editor)
        self.vertical_splitter.addWidget(self.handwriting_panel)
        self.vertical_splitter.setStretchFactor(0, 4)
        self.vertical_splitter.setStretchFactor(1, 1)
        editor_layout.addWidget(self.vertical_splitter)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(editor_container)
        splitter.setSizes([245, 1175])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        status = QStatusBar()
        self.status_label = QLabel('Ready')
        self.math_result = QLabel('')
        self.word_count_label = QLabel('0 words')
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(self.math_result)
        status.addPermanentWidget(self.word_count_label)
        self.setStatusBar(status)
        self.handwriting_panel.configure_mode('text')

    def _load_tree(self):
        self.tree.clear()
        root = QTreeWidgetItem(['Notes'])
        root.setData(0, Qt.ItemDataRole.UserRole, str(self.manager.root))
        self.tree.addTopLevelItem(root)
        self._populate_tree(root, self.manager.root)
        root.setExpanded(True)

    def _populate_tree(self, parent, folder):
        for path in self.manager.list_entries(folder):
            item = QTreeWidgetItem([path.stem if path.is_file() else path.name])
            item.setData(0, Qt.ItemDataRole.UserRole, str(path))
            parent.addChild(item)
            if path.is_dir():
                self._populate_tree(item, path)

    def _selected_path(self):
        item = self.tree.currentItem()
        return Path(item.data(0, Qt.ItemDataRole.UserRole)) if item else self.manager.root

    def _selected_folder(self):
        path = self._selected_path()
        return path if path.is_dir() else path.parent

    def _activate_item(self, item):
        path = Path(item.data(0, Qt.ItemDataRole.UserRole))
        if path.is_file():
            self._open_path(path)

    def _select_path(self, path):
        iterator = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while iterator:
            item = iterator.pop()
            if Path(item.data(0, Qt.ItemDataRole.UserRole)) == Path(path):
                self.tree.setCurrentItem(item)
                return
            iterator.extend(item.child(i) for i in range(item.childCount()))

    def _filter_tree(self):
        query = self.search.toPlainText().strip().casefold()
        root = self.tree.topLevelItem(0)
        if not root:
            return
        def visit(item):
            child_match = any(visit(item.child(i)) for i in range(item.childCount()))
            match = not query or query in item.text(0).casefold() or child_match
            item.setHidden(not match)
            return match
        visit(root)

    def _tree_menu(self, position):
        item = self.tree.itemAt(position)
        if item:
            self.tree.setCurrentItem(item)
        path = self._selected_path()
        menu = QMenu(self)
        menu.addAction('Open', lambda: self._open_path(path)).setEnabled(path.is_file())
        menu.addAction('Rename', self.rename_selected).setEnabled(path != self.manager.root)
        menu.addAction('Duplicate', self.duplicate_selected).setEnabled(path.is_file())
        menu.addAction('Move to folder…', self.move_selected).setEnabled(path.is_file())
        menu.addSeparator()
        menu.addAction('Delete', self.delete_selected).setEnabled(path != self.manager.root)
        menu.addAction('Reveal note library', self.reveal_storage)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _confirm_unsaved(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Unsaved changes', 'Save changes before continuing?', QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_note()
        return True

    def _open_path(self, path):
        if not self._confirm_unsaved():
            return
        try:
            document = self.manager.open_note(path)
        except NoteError as exc:
            self._error(str(exc))
            self.refresh()
            return
        self.document = document
        self.current_note = document.path
        self.editor.load_document(document)
        self.math_contexts.setdefault(document.path, MathEvaluator())
        self.setWindowTitle(f'{document.title} — {APP_NAME}')
        self._select_path(document.path)
        self._update_count()
        self.status_label.setText(f'Opened {document.path.name}')

    def new_note(self, _checked=False, default_name=None):
        if not self._confirm_unsaved():
            return
        name, ok = (default_name, True) if default_name else QInputDialog.getText(self, 'New note', 'Note name:')
        if not ok:
            return
        try:
            document = self.manager.create_note(name or 'Untitled', self._selected_folder())
        except NoteError as exc:
            self._error(str(exc))
            return
        self.refresh()
        self._open_path(document.path)

    def new_folder(self):
        name, ok = QInputDialog.getText(self, 'New folder', 'Folder name:')
        if not ok:
            return
        try:
            self.manager.create_folder(name, self._selected_folder())
            self.refresh()
            self.status_label.setText(f'Created folder {name}')
        except NoteError as exc:
            self._error(str(exc))

    def save_note(self):
        if not self.document:
            return False
        self._sync_document()
        try:
            self.manager.save_note(self.document)
        except NoteError as exc:
            self._error(str(exc))
            return False
        self.status_label.setText(f'Saved {self.document.path.name}')
        return True

    def save_as(self):
        if not self.document:
            return False
        path, _ = QFileDialog.getSaveFileName(self, 'Save note as', str(self.manager.root / f'{self.document.title}.md'), 'Markdown (*.md);;Text (*.txt)')
        if not path:
            return False
        self._sync_document()
        try:
            self.manager.save_note(self.document, path)
        except NoteError as exc:
            self._error(str(exc))
            return False
        self.current_note = self.document.path
        self.refresh()
        self.status_label.setText(f'Saved as {self.document.path.name}')
        return True

    def open_external(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open note', str(self.manager.root), 'Notes (*.md *.txt)')
        if path:
            self._open_path(Path(path))

    def refresh(self):
        current = self.document.path if self.document else None
        self._load_tree()
        if current:
            self._select_path(current)

    def rename_selected(self):
        path = self._selected_path()
        name, ok = QInputDialog.getText(self, 'Rename', 'New name:', text=path.stem if path.is_file() else path.name)
        if not ok:
            return
        try:
            target = self.manager.rename_note(path, name)
            if self.document and self.document.path == path:
                self.document.path = target
                self.current_note = target
            self.refresh()
        except (NoteError, OSError) as exc:
            self._error(str(exc))

    def duplicate_selected(self):
        try:
            target = self.manager.duplicate_note(self._selected_path())
            self.refresh()
            self._select_path(target)
            self.status_label.setText(f'Duplicated as {target.name}')
        except NoteError as exc:
            self._error(str(exc))

    def move_selected(self):
        folders = [self.manager.root] + [p for p in self.manager.root.rglob('*') if p.is_dir() and not p.name.startswith('.')]
        labels = ['Notes'] + [str(p.relative_to(self.manager.root)) for p in folders[1:]]
        label, ok = QInputDialog.getItem(self, 'Move note', 'Destination:', labels, 0, False)
        if not ok:
            return
        try:
            target = self.manager.move_note(self._selected_path(), folders[labels.index(label)])
            if self.document and self.document.path.name == target.name:
                self.document.path = target
                self.current_note = target
            self.refresh()
        except (NoteError, OSError) as exc:
            self._error(str(exc))

    def delete_selected(self):
        path = self._selected_path()
        if QMessageBox.question(self, 'Delete', f'Delete “{path.name}”? This cannot be undone.') != QMessageBox.StandardButton.Yes:
            return
        if self.document and (self.document.path == path or path in self.document.path.parents) and not self._confirm_unsaved():
            return
        try:
            self.manager.delete_note(path)
        except (NoteError, OSError) as exc:
            self._error(str(exc))
            return
        if self.document and not self.document.path.exists():
            self.document = None
            self.current_note = None
            notes = self.manager.files()
            if notes:
                self._open_path(notes[0])
            else:
                self.new_note(default_name='Untitled')
        self.refresh()

    def reveal_storage(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.manager.root)))

    def toggle_handwriting_panel(self):
        visible = not self.handwriting_panel.isVisible()
        self.handwriting_panel.setVisible(visible)
        if not visible:
            self.handwriting_panel.auto_timer.stop()
        self.panel_mode_button.setChecked(visible)
        if visible:
            self.vertical_splitter.setSizes([560, 280])
        self.status_label.setText('Handwriting panel opened' if visible else 'Handwriting panel closed')

    def toggle_input_mode(self):
        self.input_mode = 'math' if self.input_mode == 'text' else 'text'
        self.input_mode_button.setText('Math' if self.input_mode == 'math' else 'Text')
        self.handwriting_panel.configure_mode(self.input_mode)
        self.math_expression = ''
        model = self.math_recognizer if self.input_mode == 'math' else self.writing
        state = 'ready' if model.ready else 'unavailable'
        self.status_label.setText(f'{self.input_mode.title()} mode · recogniser {state}')

    def _text_changed(self):
        if self.document:
            self.document.text = self.editor.text.toPlainText()
            self.document.mark_dirty()
        self._update_count()

    def _sync_document(self):
        self.document.text = self.editor.text.toPlainText()

    def _autosave(self):
        if self.dirty:
            self.save_note()
            self.status_label.setText('Autosaved')

    def _recognise_panel(self, automatic=False):
        canvas = self.handwriting_panel.canvas
        if self._recognition_running:
            self.status_label.setText('Recognition is already running')
            if automatic:
                self.handwriting_panel.auto_timer.start()
            return
        if self.input_mode == 'math' and not automatic and not canvas.has_ink() and self.math_expression:
            self._insert_text_at_cursor(self.math_expression)
            self.status_label.setText(f'Inserted mathematical expression: {self.math_expression}')
            self.math_expression = ''
            self.handwriting_panel.preview.clear()
            return
        if not canvas.has_ink():
            self.status_label.setText('Write something before recognising')
            return
        if self.input_mode == 'text' and not automatic and self._writing_preview_revision == canvas.revision:
            text = self.handwriting_panel.preview.toPlainText().strip()
            if text:
                self._insert_text_at_cursor(text)
                self.status_label.setText('Recognition preview inserted; original ink retained')
                return
        if self.input_mode == 'math':
            self._recognise_math_symbol(automatic)
            return
        if not self.writing.ready:
            if not automatic:
                self._error(self.writing.error or 'Writing CNN unavailable.')
            return
        self._recognition_running = True
        self.handwriting_panel.set_busy(True)
        worker = RecognitionWorker(self.writing, canvas.to_pil(), self.handwriting_panel, canvas, self.handwriting_panel.preview, automatic, canvas.revision)
        worker.signals.finished.connect(self._writing_done)
        worker.signals.error.connect(self._writing_error)
        self.thread_pool.start(worker)

    def _writing_done(self, result, panel, canvas, info, automatic, revision):
        self._recognition_running = False
        if canvas.revision != revision:
            info.setPlainText('The ink changed during recognition. Run recognition again.')
            panel_widget = getattr(self, 'handwriting_panel', None)
            if panel_widget is not None:
                panel_widget.auto_timer.start()
            return
        text = info.toPlainText().strip() if self._writing_preview_revision == revision else result.text
        info.setReadOnly(False)
        info.setPlainText(text)
        self._writing_preview = result
        self._writing_preview_revision = revision
        if automatic:
            self.status_label.setText('Automatic recognition preview ready')
        elif text:
            self._insert_text_at_cursor(text)
            self.status_label.setText('Recognition complete — text inserted; original ink retained')

    def _recognise_math_symbol(self, automatic):
        canvas = self.handwriting_panel.canvas
        if not self.math_recognizer.ready:
            if not automatic:
                self._error(self.math_recognizer.error or 'Math CNN unavailable.')
            return
        try:
            candidates = self.math_recognizer.predict_single(canvas.to_pil(), top_k=3)
        except Exception as exc:
            self.handwriting_panel.preview.setPlainText(f'Math recognition failed: {exc}')
            self.status_label.setText('Math recognition failed')
            return
        if not candidates:
            return
        prediction, confidence = candidates[0]
        token = self.math_recognizer.replacements.get(prediction, prediction)
        token = token.replace('\\', '').replace('{}', '').strip()
        self.math_expression += token
        self.handwriting_panel.preview.setPlainText(f'{self.math_expression}   ·   last symbol {confidence:.0%}')
        self.status_label.setText(f'Expression: {self.math_expression}')
        self.handwriting_panel.auto_timer.stop()
        canvas.clear()
        self.handwriting_panel.auto_timer.stop()
        if not automatic:
            self._insert_text_at_cursor(self.math_expression)
            self.status_label.setText(f'Inserted mathematical expression: {self.math_expression}')
            self.math_expression = ''
            self.handwriting_panel.preview.clear()

    def _clear_math_expression(self):
        if self.input_mode == 'math':
            self.math_expression = ''

    def _writing_error(self, error, panel, canvas, info, automatic, revision):
        self._recognition_running = False
        info.setReadOnly(False)
        info.setPlainText(f'Recognition failed: {error}')
        self.status_label.setText('Recognition failed')

    def _insert_text_at_cursor(self, text):
        cursor = self.editor.text.textCursor()
        cursor.insertText(text)
        self.editor.text.setTextCursor(cursor)
        self.editor.text.setFocus()

    def _inspect_panel(self):
        canvas = self.handwriting_panel.canvas
        if not canvas.has_ink():
            self.status_label.setText('Write something before inspecting')
            return
        try:
            started = time.perf_counter()
            segmented = self.writing.segment(canvas.to_pil())
            words = sum(len(line) for line in segmented)
            chars = sum(len(word) for line in segmented for word in line)
            self.handwriting_panel.preview.setPlainText(f'{len(segmented)} lines · {words} words · {chars} characters · {(time.perf_counter() - started) * 1000:.0f} ms')
        except Exception as exc:
            self.handwriting_panel.preview.setPlainText(f'Segmentation failed: {exc}')

    def _insert_panel_as_image(self):
        canvas = self.handwriting_panel.canvas
        if not canvas.has_ink() or not self.document:
            self.status_label.setText('There is no ink to keep')
            return
        path = self.manager.handwriting / f'ink-{int(time.time() * 1000)}.png'
        canvas.export_png(path)
        relative = path.relative_to(self.manager.root).as_posix()
        self._insert_text_at_cursor(f'![Handwritten note]({relative})')
        self.status_label.setText('Handwriting image inserted into note')

    def _expression_at_cursor(self):
        cursor = self.editor.text.textCursor()
        if cursor.hasSelection():
            return cursor.selectedText().replace('\u2029', '\n').strip()
        return cursor.block().text().strip()

    def evaluate_expression(self, insert=True):
        if not self.document:
            return None
        expression = self._expression_at_cursor()
        evaluator = self.math_contexts.setdefault(self.document.path, MathEvaluator())
        try:
            result = evaluator.evaluate(expression)
        except MathEvaluationError as exc:
            self.math_result.setText('Math error')
            self.status_label.setText(str(exc))
            return None
        self.math_result.setText(f'{result.expression} → {result.result}')
        self.status_label.setText(f'{result.assignment or "Result"}: {result.result}')
        if insert:
            cursor = self.editor.text.textCursor()
            separator = ' → ' if re.search(r'=\s*\?', expression) else ' = '
            rendered = f'{expression}{separator}{result.result}'
            if cursor.hasSelection():
                cursor.insertText(rendered)
            else:
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
                cursor.insertText(f'{separator}{result.result}')
            self.editor.text.setTextCursor(cursor)
            if self.save_note():
                self.status_label.setText(f'Evaluated and saved: {rendered}')
        return result

    def insert_math_result(self):
        self.evaluate_expression(insert=True)

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    def show_settings(self):
        dialog = ShortcutSettingsDialog(self, self.actions)
        if dialog.exec():
            for key, action in self.actions.items():
                shortcut = load_shortcut(key)
                if shortcut:
                    action.setShortcut(QKeySequence(shortcut))

    def _update_count(self):
        words = len(re.findall(r'\b\w+\b', self.editor.text.toPlainText()))
        self.word_count_label.setText(f'{words} words')

    def _error(self, message):
        self.status_label.setText(message)
        QMessageBox.warning(self, APP_NAME, message)

    def closeEvent(self, event):
        self.autosave_timer.stop()
        event.accept() if self._confirm_unsaved() else event.ignore()


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
