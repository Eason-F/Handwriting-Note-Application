APP_NAME = 'InkNote'
NOTES_DIR = 'notes'
CHECKPOINT_PATH = 'checkpoints/writing_cnn.pt'
MATH_CHECKPOINT_PATH = 'checkpoints/math_cnn.pt'
AUTOSAVE_MS = 900

DARK = '#282b2e'
PANEL = '#30343a'
PANEL_2 = '#3a3f45'
EDITOR = '#2c3034'
TEXT = '#e4e7e2'
MUTED = '#a5aca5'
ACCENT = '#8eae9d'
ACCENT_SOFT = '#40554c'
BORDER = '#444a50'
WHITE = '#f2f4f0'

STYLESHEET = f'''
QMainWindow, QWidget {{ background: {DARK}; color: {TEXT}; }}
QFrame#sidePanel, QDockWidget {{ background: {PANEL}; border: 0; }}
QFrame#editorPanel {{ background: {EDITOR}; }}
QFrame#rightPanel {{ background: {PANEL}; }}
QLabel#appTitle {{ color: {WHITE}; font-size: 15px; font-weight: 700; padding: 5px 8px; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 10px; font-weight: 700; padding: 8px 8px 4px; letter-spacing: 1px; }}
QLabel#status {{ color: {MUTED}; font-size: 11px; padding: 6px 10px; }}
QLineEdit, QTextEdit, QListWidget, QComboBox {{
    background: {EDITOR}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px;
}}
QLineEdit {{ padding: 7px 9px; }}
QTextEdit#editor {{ border: 0; border-radius: 0; padding: 26px 64px; font-size: 16px; selection-background-color: {ACCENT_SOFT}; }}
QListWidget {{ border: 0; padding: 4px; }}
QListWidget::item {{ padding: 8px 10px; border-radius: 5px; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {WHITE}; }}
QPushButton {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px 11px; }}
QPushButton:hover {{ background: #444a50; border-color: #5b6269; }}
QPushButton#primary {{ background: {ACCENT}; color: #1e2822; border: 0; font-weight: 700; }}
QPushButton#primary:hover {{ background: #a3c0b1; }}
QToolBar {{ background: {PANEL}; border: 0; spacing: 5px; padding: 5px 7px; }}
QToolButton {{ color: {TEXT}; background: transparent; border-radius: 5px; padding: 6px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: #363b40; color: {MUTED}; border: 1px solid transparent; border-radius: 6px; padding: 7px 13px; margin: 1px 2px; min-width: 70px; }}
QTabBar::tab:hover {{ background: #40464b; color: {TEXT}; }}
QTabBar::tab:selected {{ background: {EDITOR}; color: {WHITE}; border-color: {BORDER}; }}
QToolButton:hover {{ background: {PANEL_2}; }}
QTabWidget::pane {{ border: 0; background: {EDITOR}; }}
QTabBar {{ background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 9px 14px; border: 0; }}
QTabBar::tab:selected {{ color: {WHITE}; background: {EDITOR}; }}
QPushButton#modeToggle {{
    background: {PANEL_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 70px;
}}
QPushButton#modeToggle:hover {{ background: #444a50; border-color: #5b6269; }}
QPushButton#modeToggle:checked {{
    background: {ACCENT};
    color: #1e2822;
    border-color: {ACCENT};
    font-weight: 700;
}}
QPushButton#modeToggle:checked:hover {{ background: #a3c0b1; border-color: #a3c0b1; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
'''
