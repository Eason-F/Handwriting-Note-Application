from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QKeySequenceEdit, QPushButton, QVBoxLayout, QWidget


DEFAULT_SHORTCUTS = {
    'new': 'Ctrl+N',
    'open': 'Ctrl+O',
    'save': 'Ctrl+S',
    'close': 'Ctrl+W',
    'search': 'Ctrl+F',
    'handwriting': 'Ctrl+Shift+H',
    'calculate': 'Ctrl+=',
    'edit': 'Ctrl+E',
    'settings': 'Ctrl+,',
}


class ShortcutSettingsDialog(QDialog):
    def __init__(self, parent, actions):
        super().__init__(parent)
        self.setWindowTitle('Settings — Shortcuts')
        self.setMinimumWidth(430)
        self.edits = {}
        layout = QVBoxLayout(self)
        hint = QLabel('Change keyboard shortcuts used by InkNote. Changes are saved for future launches.')
        hint.setWordWrap(True); layout.addWidget(hint)
        form = QFormLayout(); layout.addLayout(form)
        for key, action in actions.items():
            edit = QKeySequenceEdit(QKeySequence(action.shortcut()))
            self.edits[key] = edit
            form.addRow(action.text(), edit)
        reset = QPushButton('Reset to defaults'); reset.clicked.connect(self.reset_defaults); form.addRow('', reset)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self.save); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def reset_defaults(self):
        for key, edit in self.edits.items(): edit.setKeySequence(QKeySequence(DEFAULT_SHORTCUTS[key]))

    def save(self):
        values = {key: edit.keySequence().toString() for key, edit in self.edits.items() if edit.keySequence().toString()}
        seen = {}
        for key, value in values.items():
            if value in seen:
                QMessageBox.warning(self, 'Shortcut conflict', f'“{value}” is assigned to both “{seen[value]}” and “{self._label(key)}”.')
                return
            seen[value] = self._label(key)
        settings = QSettings('InkNote', 'InkNote')
        for key, edit in self.edits.items(): settings.setValue(f'shortcuts/{key}', edit.keySequence().toString())
        self.accept()

    def _label(self, key):
        return self.parent().actions[key].text() if self.parent() and key in self.parent().actions else key


def load_shortcut(key):
    value = QSettings('InkNote', 'InkNote').value(f'shortcuts/{key}', DEFAULT_SHORTCUTS.get(key, ''), type=str)
    return value or ''
