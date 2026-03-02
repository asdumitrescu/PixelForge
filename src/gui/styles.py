"""Dark theme QSS stylesheet for PixelForge."""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Ubuntu", "Cantarell", sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}

QMenuBar::item:selected {
    background-color: #313244;
}

QMenu {
    background-color: #1e1e2e;
    border: 1px solid #313244;
}

QMenu::item:selected {
    background-color: #313244;
}

QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 4px;
    padding: 2px;
}

QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
}

QLabel {
    color: #cdd6f4;
}

QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 10px;
    color: #cdd6f4;
    min-height: 24px;
}

QComboBox:hover {
    border-color: #7c3aed;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #7c3aed;
    color: #cdd6f4;
}

QSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px;
    color: #cdd6f4;
    min-height: 24px;
}

QSpinBox:hover {
    border-color: #7c3aed;
}

QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #7c3aed;
    border-color: #7c3aed;
}

QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #a6adc8;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""

BUTTON_PRIMARY = """
QPushButton {
    background-color: #7c3aed;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 15px;
    font-weight: bold;
    min-height: 36px;
}

QPushButton:hover {
    background-color: #8b5cf6;
}

QPushButton:pressed {
    background-color: #6d28d9;
}

QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
"""

BUTTON_SECONDARY = """
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    min-height: 30px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #7c3aed;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}
"""

BUTTON_DANGER = """
QPushButton {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #f5a0b8;
}
"""

PROGRESS_BAR = """
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    min-height: 22px;
    font-size: 12px;
}

QProgressBar::chunk {
    background-color: #7c3aed;
    border-radius: 4px;
}
"""

DROP_ZONE = """
QLabel {
    background-color: #181825;
    border: 2px dashed #45475a;
    border-radius: 12px;
    color: #6c7086;
    font-size: 16px;
    padding: 40px;
}

QLabel:hover {
    border-color: #7c3aed;
    color: #a6adc8;
}
"""

DROP_ZONE_ACTIVE = """
QLabel {
    background-color: #1e1e3e;
    border: 2px dashed #7c3aed;
    border-radius: 12px;
    color: #7c3aed;
    font-size: 16px;
    padding: 40px;
}
"""
