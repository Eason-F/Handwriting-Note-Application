from pathlib import Path


APP_NAME = 'InkNote'
NOTES_DIR = 'notes'
RESOURCE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = RESOURCE_ROOT / 'checkpoints/writing_cnn.pt'
MATH_CHECKPOINT_PATH = RESOURCE_ROOT / 'checkpoints/math_cnn.pt'
AUTOSAVE_MS = 900

# Visual system
WINDOW = '#171a1d'
SIDEBAR = '#1d2125'
SURFACE = '#22272c'
SURFACE_RAISED = '#292f35'
EDITOR = '#1b1f23'
TEXT = '#edf1ed'
MUTED = '#9aa49d'
FAINT = '#747e77'
ACCENT = '#8fb9a4'
ACCENT_HOVER = '#a4cab7'
ACCENT_SOFT = '#31483e'
BORDER = '#343b41'
BORDER_STRONG = '#485159'

STYLESHEET = f'''
* {{ font-family: "Inter", "SF Pro Text", "Segoe UI", sans-serif; font-size: 13px; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {WINDOW}; }}
QWidget#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
QWidget#editorArea, QWidget#editorWidget, QScrollArea#adaptiveEditor, QWidget#editorContent {{ background: {EDITOR}; }}
QFrame#handwritingPanel {{ background: {SURFACE}; border: 0; border-top: 1px solid {BORDER_STRONG}; }}
QFrame#handwritingPanel[busy="true"] {{ border-top: 2px solid {ACCENT}; }}
QLabel {{ background: transparent; }}
QLabel#appTitle {{ color: {TEXT}; font-size: 17px; font-weight: 700; padding: 4px 2px 10px 2px; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; padding: 2px 0; }}
QLabel#statusText, QLabel#wordCount, QLabel#mathResult {{ color: {MUTED}; font-size: 12px; }}
QLabel#mathResult {{ color: {ACCENT}; padding: 0 10px; }}
QLineEdit, QTextEdit, QComboBox, QKeySequenceEdit {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 7px 10px; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QLineEdit:hover, QTextEdit:hover, QComboBox:hover, QKeySequenceEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QKeySequenceEdit:focus {{ border: 1px solid {ACCENT}; }}
QTextEdit#searchInput {{ background: {SURFACE}; border-radius: 9px; padding: 7px 10px; }}
QTextEdit#editor {{
    background: {EDITOR}; border: 0; border-radius: 0; padding: 36px 64px;
    font-family: "SF Pro Text", "Segoe UI", sans-serif; font-size: 16px;
}}
QTextEdit#activeMarkdownSection {{
    background: #20252a; border: 1px solid {BORDER_STRONG}; border-left: 3px solid {ACCENT};
    border-radius: 9px; padding: 16px 18px; font-family: "SF Mono", "Consolas", monospace;
    font-size: 14px; line-height: 1.35;
}}
QTextEdit#activeMarkdownSection:focus {{ border: 1px solid {ACCENT}; border-left: 3px solid {ACCENT}; }}
QTextBrowser#renderedSection {{
    background: transparent; color: {TEXT}; border: 1px solid transparent; border-radius: 9px;
    padding: 12px 20px; font-size: 15px;
}}
QTextBrowser#renderedSection:hover {{ background: {SURFACE}; border-color: {BORDER}; }}
QTextEdit#recognitionPreview {{ background: {SURFACE_RAISED}; border-radius: 8px; padding: 7px 10px; font-size: 13px; }}
QTreeWidget#noteTree {{ background: transparent; border: 0; outline: 0; padding: 4px 0; }}
QTreeWidget#noteTree::item {{ min-height: 31px; padding: 3px 8px; margin: 1px 0; border-radius: 7px; color: {MUTED}; }}
QTreeWidget#noteTree::item:hover {{ background: {SURFACE_RAISED}; color: {TEXT}; }}
QTreeWidget#noteTree::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; font-weight: 600; }}
QTreeWidget#noteTree::branch {{ background: transparent; }}
QTreeWidget#noteTree::branch:selected {{ background: transparent; }}
QPushButton {{
    min-height: 32px; background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 0 12px;
}}
QPushButton:hover {{ background: #323940; border-color: {BORDER_STRONG}; }}
QPushButton:pressed {{ background: #20262a; border-color: {ACCENT}; padding-top: 1px; }}
QPushButton:focus {{ border: 1px solid {ACCENT}; }}
QPushButton:checked, QPushButton#modeButton {{ background: {ACCENT_SOFT}; border-color: #557565; font-weight: 600; }}
QPushButton:disabled {{ background: {SURFACE}; color: {FAINT}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; color: #142119; border: 0; font-weight: 700; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: #79a58f; }}
QPushButton#compactButton {{ min-width: 34px; max-width: 34px; padding: 0; font-size: 15px; }}
QToolBar#mainToolbar {{ background: {SIDEBAR}; border: 0; border-bottom: 1px solid {BORDER}; spacing: 5px; padding: 7px 10px; }}
QToolBar#mainToolbar::separator {{ background: {BORDER}; width: 1px; margin: 7px 8px; }}
QToolButton {{ min-height: 30px; background: transparent; color: {MUTED}; border: 0; border-radius: 7px; padding: 1px 9px; }}
QToolButton:hover {{ background: {SURFACE_RAISED}; color: {TEXT}; }}
QToolButton:pressed {{ background: {ACCENT_SOFT}; }}
QToolButton:disabled {{ color: {FAINT}; }}
QStatusBar {{ background: {SIDEBAR}; border-top: 1px solid {BORDER}; padding: 2px 8px; }}
QStatusBar::item {{ border: 0; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QCheckBox {{ color: {MUTED}; spacing: 7px; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {BORDER_STRONG}; border-radius: 4px; background: {EDITOR}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; border: 2px solid {SURFACE}; width: 14px; margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
QMenu {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER_STRONG}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 7px 26px 7px 10px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu::item:disabled {{ color: {FAINT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
QTabBar {{ background: {SIDEBAR}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 9px 16px; border: 0; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; background: {SURFACE}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; font-weight: 600; }}
QTabWidget::pane {{ background: {EDITOR}; border: 0; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; min-height: 28px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: #5a646c; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; min-width: 28px; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QToolTip {{ background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 5px; }}
'''
