"""Main application window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from src.constants import APP_NAME, APP_VERSION


class MainWindow(QMainWindow):
    """PixelForge main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(900, 600)

        # Placeholder — replaced in Phase 4 with full layout
        central = QWidget()
        layout = QVBoxLayout(central)
        label = QLabel(f"{APP_NAME} — AI Image Super-Resolution")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #7c3aed; padding: 40px;")
        layout.addWidget(label)

        status_label = QLabel("Ready — drop an image to begin")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(status_label)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
