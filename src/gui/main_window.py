"""Main application window — wires GUI, engine, and workers together."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from src.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_TILE_PAD,
    DEFAULT_TILE_SIZE,
    MODELS_DIR,
    OUTPUT_FILE_FILTER,
)
from src.engine.device import DeviceManager
from src.engine.image_utils import load_image, save_image
from src.engine.model_downloader import ModelDownloader
from src.engine.model_manager import ModelManager
from src.engine.model_registry import get_model_entry
from src.engine.upscaler import Upscaler
from src.gui.controls_panel import ControlsPanel
from src.gui.download_dialog import DownloadDialog
from src.gui.drop_zone import DropZone
from src.gui.image_viewer import ImageViewer
from src.gui.qt_utils import format_dimensions, format_file_size, numpy_to_qpixmap
from src.gui.styles import DARK_THEME
from src.workers.model_load_worker import ModelLoadWorker
from src.workers.upscale_worker import UpscaleWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """PixelForge main application window.

    Layout: ImageViewer (left, stretches) | ControlsPanel (right, fixed 280px)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(DARK_THEME)

        # Engine components
        self._device_manager = DeviceManager()
        self._model_manager = ModelManager(MODELS_DIR, self._device_manager)
        self._model_downloader = ModelDownloader(MODELS_DIR)
        self._upscaler = Upscaler(
            self._model_manager, self._device_manager,
            DEFAULT_TILE_SIZE, DEFAULT_TILE_PAD,
        )

        # State
        self._input_image_path: Path | None = None
        self._input_image: np.ndarray | None = None
        self._input_metadata: dict | None = None
        self._output_image: np.ndarray | None = None
        self._upscale_worker: UpscaleWorker | None = None
        self._model_load_worker: ModelLoadWorker | None = None

        self._setup_ui()
        self._setup_menubar()
        self._setup_connections()
        self._update_device_info()

    def _setup_ui(self) -> None:
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: stacked widget (drop zone / image viewer)
        self._stack = QStackedWidget()
        self._drop_zone = DropZone()
        self._image_viewer = ImageViewer()
        self._stack.addWidget(self._drop_zone)
        self._stack.addWidget(self._image_viewer)
        self._stack.setCurrentWidget(self._drop_zone)

        main_layout.addWidget(self._stack, stretch=1)

        # Right: controls panel
        self._controls = ControlsPanel()
        main_layout.addWidget(self._controls)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready — drag & drop an image or click to browse")

    def _setup_menubar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_file_open)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Result As...", self)
        save_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence("+"))
        zoom_in_action.triggered.connect(self._image_viewer.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence("-"))
        zoom_out_action.triggered.connect(self._image_viewer.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("&Fit to View", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self._image_viewer.fit_to_view)
        view_menu.addAction(fit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_connections(self) -> None:
        self._drop_zone.file_dropped.connect(self._on_file_loaded)
        self._controls.upscale_requested.connect(self._on_upscale_requested)
        self._controls.cancel_requested.connect(self._on_cancel)
        self._controls.save_requested.connect(self._on_save)
        self._controls.tile_size_changed.connect(self._on_tile_size_changed)

    # --- File Handling ---

    def _on_file_open(self) -> None:
        from src.constants import INPUT_FILE_FILTER

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", INPUT_FILE_FILTER
        )
        if file_path:
            self._on_file_loaded(file_path)

    def _on_file_loaded(self, file_path: str) -> None:
        try:
            path = Path(file_path)
            image, metadata = load_image(path)

            self._input_image_path = path
            self._input_image = image
            self._input_metadata = metadata
            self._output_image = None

            # Display in viewer
            pixmap = numpy_to_qpixmap(image)
            self._image_viewer.set_image(pixmap)
            self._stack.setCurrentWidget(self._image_viewer)

            # Update status
            h, w = image.shape[:2]
            size = path.stat().st_size
            self.statusBar().showMessage(
                f"{path.name} | {format_dimensions(w, h)} | {format_file_size(size)}"
            )
            self._controls.set_save_enabled(False)
            self._controls.set_status(f"Input: {format_dimensions(w, h)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    # --- Upscaling ---

    def _on_upscale_requested(self) -> None:
        if self._input_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        model_id = self._controls.get_selected_model_id()
        entry = get_model_entry(model_id)
        if entry is None:
            QMessageBox.critical(self, "Error", f"Unknown model: {model_id}")
            return

        # Check if model is downloaded
        if not self._model_downloader.is_downloaded(entry):
            dialog = DownloadDialog(self._model_downloader, entry, self)
            if dialog.exec() != DownloadDialog.DialogCode.Accepted:
                return  # User cancelled download

        # Load model if not already loaded (or different model selected)
        model_path = self._model_downloader.get_model_path(entry)
        if (
            not self._model_manager.is_loaded()
            or self._model_manager.current_model_path != model_path
        ):
            self._controls.set_status("Loading model...")
            self._controls.set_processing(True)

            self._model_load_worker = ModelLoadWorker(
                self._model_manager, model_path, self._controls.get_use_half()
            )
            self._model_load_worker.finished.connect(self._on_model_loaded)
            self._model_load_worker.error.connect(self._on_model_load_error)
            self._model_load_worker.start()
            return

        # Model already loaded — start upscaling directly
        self._start_upscaling()

    def _on_model_loaded(self) -> None:
        self._controls.set_status("Model loaded")
        self._start_upscaling()

    def _on_model_load_error(self, msg: str) -> None:
        self._controls.set_processing(False)
        QMessageBox.critical(self, "Model Error", f"Failed to load model:\n{msg}")

    def _start_upscaling(self) -> None:
        if self._input_image is None:
            return

        self._upscaler.tile_size = self._controls.get_tile_size()
        self._controls.set_processing(True)
        self._controls.set_status("Upscaling...")

        self._upscale_worker = UpscaleWorker(self._upscaler, self._input_image)
        self._upscale_worker.progress.connect(self._on_upscale_progress)
        self._upscale_worker.finished.connect(self._on_upscale_finished)
        self._upscale_worker.error.connect(self._on_upscale_error)
        self._upscale_worker.start()

    def _on_upscale_progress(self, current: int, total: int, eta: float) -> None:
        self._controls.set_progress(current, total, eta)

    def _on_upscale_finished(self, result: object) -> None:
        result_arr = result  # It's actually np.ndarray, passed as object via Signal
        self._output_image = result_arr
        self._controls.set_processing(False)
        self._controls.set_save_enabled(True)

        # Display result
        pixmap = numpy_to_qpixmap(result_arr)
        self._image_viewer.set_image(pixmap)

        h, w = result_arr.shape[:2]
        self._controls.set_status(f"Done! Output: {format_dimensions(w, h)}")
        self.statusBar().showMessage(f"Upscaled to {format_dimensions(w, h)}")

    def _on_upscale_error(self, msg: str) -> None:
        self._controls.set_processing(False)
        if msg != "Cancelled":
            QMessageBox.critical(self, "Upscale Error", f"Upscaling failed:\n{msg}")
        else:
            self._controls.set_status("Cancelled")
            self.statusBar().showMessage("Upscaling cancelled")

    # --- Cancel ---

    def _on_cancel(self) -> None:
        if self._upscale_worker and self._upscale_worker.isRunning():
            self._upscale_worker.cancel()

    # --- Save ---

    def _on_save(self) -> None:
        if self._output_image is None:
            QMessageBox.warning(self, "Nothing to Save", "Upscale an image first.")
            return

        default_name = ""
        if self._input_image_path:
            default_name = f"{self._input_image_path.stem}_upscaled.png"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Result", default_name, OUTPUT_FILE_FILTER
        )
        if not file_path:
            return

        # Determine format from filter/extension
        path = Path(file_path)
        ext = path.suffix.lower()
        fmt_map = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WebP"}
        fmt = fmt_map.get(ext, "PNG")

        try:
            save_image(self._output_image, path, fmt=fmt, metadata=self._input_metadata)
            size = path.stat().st_size
            self.statusBar().showMessage(
                f"Saved: {path.name} ({format_file_size(size)})"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

    # --- Settings ---

    def _on_tile_size_changed(self, value: int) -> None:
        self._upscaler.tile_size = value

    # --- Info ---

    def _update_device_info(self) -> None:
        info = self._device_manager.get_device_info()
        if info.is_cuda:
            text = f"{info.name} ({info.vram_free_mb} MB free)"
        else:
            text = "CPU (no GPU detected)"
        self._controls.set_device_info(text)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
            f"<p>AI-powered image super-resolution.</p>"
            f"<p>Upscale low-resolution images to 4K+ using Real-ESRGAN, "
            f"HAT, SwinIR, and other state-of-the-art models.</p>"
            f"<p>Runs locally on your GPU — no cloud required.</p>",
        )
