"""PixelForge entry point."""

import sys

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> None:
    """Launch the PixelForge application."""
    app = QApplication(sys.argv)
    app.setApplicationName("PixelForge")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
