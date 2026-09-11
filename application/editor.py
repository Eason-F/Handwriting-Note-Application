from __future__ import annotations

import re

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QTextBrowser, QTextEdit, QVBoxLayout, QWidget


class MarkdownSectionView(QTextBrowser):
    activated = Signal()

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.setObjectName('renderedSection')
        self.setOpenExternalLinks(False)
        self.setMarkdown(source)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.document().documentLayout().documentSizeChanged.connect(self._fit_height)
        self._fit_height()

    def _fit_height(self, *_):
        height = int(self.document().size().height() + self.contentsMargins().top() + self.contentsMargins().bottom() + 18)
        self.setFixedHeight(max(54, height))

    def mousePressEvent(self, event):
        self.activated.emit()
        event.accept()


class AdaptiveMarkdownEditor(QScrollArea):
    """Shows one Markdown section as source while rendering all other sections."""

    textChanged = Signal()
    HEADING = re.compile(r'(?m)^#{1,6}[ \t]+.+(?:\n|$)')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('adaptiveEditor')
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._sections = ['']
        self._active_index = 0
        self._active_editor = None
        self._preview_mode = False
        self._placeholder = 'Start writing…'

        self.content = QWidget()
        self.content.setObjectName('editorContent')
        self.sections_layout = QVBoxLayout(self.content)
        self.sections_layout.setContentsMargins(54, 28, 54, 40)
        self.sections_layout.setSpacing(8)
        self.sections_layout.addStretch()
        self.setWidget(self.content)
        self._rebuild()

    @classmethod
    def split_sections(cls, text):
        matches = list(cls.HEADING.finditer(text))
        if not matches:
            return [text]
        starts = [match.start() for match in matches]
        if starts[0] != 0:
            starts.insert(0, 0)
        starts.append(len(text))
        return [text[starts[index]:starts[index + 1]] for index in range(len(starts) - 1)]

    def setPlainText(self, text):
        self._sections = self.split_sections(text)
        self._active_index = min(self._active_index, len(self._sections) - 1)
        self._rebuild()

    def toPlainText(self):
        self._commit_active()
        return ''.join(self._sections)

    def setPlaceholderText(self, text):
        self._placeholder = text
        if self._active_editor:
            self._active_editor.setPlaceholderText(text)

    def setAcceptRichText(self, _enabled):
        pass

    def textCursor(self):
        if self._active_editor is None:
            self.set_preview(False)
        return self._active_editor.textCursor()

    def setTextCursor(self, cursor):
        if self._active_editor is None:
            self.set_preview(False)
        if cursor.document() == self._active_editor.document():
            self._active_editor.setTextCursor(cursor)

    def setFocus(self, reason=Qt.FocusReason.OtherFocusReason):
        if self._active_editor is None:
            self.set_preview(False)
        self._active_editor.setFocus(reason)

    def undo(self):
        if self._active_editor is None:
            self.set_preview(False)
        self._active_editor.undo()

    def redo(self):
        if self._active_editor is None:
            self.set_preview(False)
        self._active_editor.redo()

    def document(self):
        if self._active_editor is None:
            self.set_preview(False)
        return self._active_editor.document()

    def set_preview(self, enabled):
        enabled = bool(enabled)
        if enabled == self._preview_mode:
            return
        self._commit_active()
        self._preview_mode = enabled
        self._rebuild()

    def is_preview(self):
        return self._preview_mode

    def _commit_active(self):
        if self._active_editor is not None:
            self._sections[self._active_index] = self._active_editor.toPlainText()

    def _activate(self, index):
        if self._preview_mode:
            self._preview_mode = False
            self._active_index = index
            self._rebuild()
            self._active_editor.setFocus()
            return
        if index == self._active_index:
            self._active_editor.setFocus()
            return
        self._commit_active()
        full_text = ''.join(self._sections)
        sections = self.split_sections(full_text)
        self._sections = sections
        self._active_index = min(index, len(sections) - 1)
        self._rebuild()
        cursor = self._active_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._active_editor.setTextCursor(cursor)
        self._active_editor.setFocus()

    def _raw_changed(self):
        self._sections[self._active_index] = self._active_editor.toPlainText()
        self._resize_active()
        self.textChanged.emit()

    def _resize_active(self):
        document_height = int(self._active_editor.document().size().height() + 24)
        self._active_editor.setMinimumHeight(max(90, document_height))

    def _rebuild(self):
        while self.sections_layout.count() > 1:
            item = self.sections_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self._active_editor = None
        for index, source in enumerate(self._sections):
            if index == self._active_index and not self._preview_mode:
                editor = QTextEdit()
                editor.setObjectName('activeMarkdownSection')
                editor.setAcceptRichText(False)
                editor.setPlaceholderText(self._placeholder)
                editor.setPlainText(source)
                editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                editor.textChanged.connect(self._raw_changed)
                self._active_editor = editor
                self.sections_layout.insertWidget(index, editor)
                self._resize_active()
            else:
                viewer = MarkdownSectionView(source)
                viewer.activated.connect(lambda section=index: self._activate(section))
                self.sections_layout.insertWidget(index, viewer)
