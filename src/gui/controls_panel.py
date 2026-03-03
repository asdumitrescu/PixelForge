"""Right-side controls panel with model selection, settings, and action buttons."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.constants import DEFAULT_TILE_SIZE, MIN_TILE_SIZE
from src.engine.model_registry import MODEL_REGISTRY
from src.gui.qt_utils import format_eta
from src.gui.styles import BUTTON_DANGER, BUTTON_PRIMARY, BUTTON_SECONDARY, PROGRESS_BAR


class ControlsPanel(QWidget):
    """Side panel with model selection, inference settings, and action buttons."""

    upscale_requested = Signal()
    cancel_requested = Signal()
    save_requested = Signal()
    model_changed = Signal(str)
    tile_size_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Model Selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        for model_id, entry in MODEL_REGISTRY.items():
            self._model_combo.addItem(entry.display_name, userData=model_id)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self._model_combo)
        self._model_desc = QLabel("")
        self._model_desc.setWordWrap(True)
        self._model_desc.setStyleSheet("color: #a6adc8; font-size: 11px;")
        model_layout.addWidget(self._model_desc)
        self._on_model_changed()
        layout.addWidget(model_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Tile Size:"))
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(MIN_TILE_SIZE, 512)
        self._tile_spin.setSingleStep(32)
        self._tile_spin.setValue(DEFAULT_TILE_SIZE)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.valueChanged.connect(self.tile_size_changed)
        tile_row.addWidget(self._tile_spin)
        settings_layout.addLayout(tile_row)
        self._half_check = QCheckBox("Use half precision (fp16)")
        self._half_check.setChecked(True)
        self._half_check.setToolTip("Reduces VRAM usage by ~50%. Recommended for 4GB GPUs.")
        settings_layout.addWidget(self._half_check)
        self._face_enhance_check = QCheckBox("Enhance faces (GFPGAN)")
        self._face_enhance_check.setChecked(False)
        self._face_enhance_check.setToolTip(
            "Detect and restore faces after upscaling. Requires GFPGAN model (~349 MB)."
        )
        settings_layout.addWidget(self._face_enhance_check)
        layout.addWidget(settings_group)

        # Actions
        self._upscale_btn = QPushButton("Upscale")
        self._upscale_btn.setStyleSheet(BUTTON_PRIMARY)
        self._upscale_btn.clicked.connect(self.upscale_requested)
        layout.addWidget(self._upscale_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(BUTTON_DANGER)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.hide()
        layout.addWidget(self._cancel_btn)

        self._save_btn = QPushButton("Save Result")
        self._save_btn.setStyleSheet(BUTTON_SECONDARY)
        self._save_btn.clicked.connect(self.save_requested)
        self._save_btn.setEnabled(False)
        layout.addWidget(self._save_btn)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setStyleSheet(PROGRESS_BAR)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)
        self._eta_label = QLabel("")
        self._eta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eta_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        self._eta_label.hide()
        layout.addWidget(self._eta_label)

        # Device Info
        self._device_label = QLabel("Device: detecting...")
        self._device_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self._device_label)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self._status_label)
        layout.addStretch()

    def get_selected_model_id(self) -> str:
        return self._model_combo.currentData()

    def get_tile_size(self) -> int:
        return self._tile_spin.value()

    def get_use_half(self) -> bool:
        return self._half_check.isChecked()

    def get_enhance_faces(self) -> bool:
        return self._face_enhance_check.isChecked()

    def set_progress(self, current: int, total: int, eta: float) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        pct = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setFormat(f"{pct}% ({current}/{total} tiles)")
        self._eta_label.setText(f"ETA: {format_eta(eta)}")

    def set_processing(self, active: bool) -> None:
        self._upscale_btn.setVisible(not active)
        self._upscale_btn.setEnabled(not active)
        self._cancel_btn.setVisible(active)
        self._progress_bar.setVisible(active)
        self._eta_label.setVisible(active)
        self._model_combo.setEnabled(not active)
        self._tile_spin.setEnabled(not active)
        self._half_check.setEnabled(not active)
        self._face_enhance_check.setEnabled(not active)
        if active:
            self._progress_bar.setValue(0)
            self._eta_label.setText("ETA: calculating...")
            self._save_btn.setEnabled(False)

    def set_save_enabled(self, enabled: bool) -> None:
        self._save_btn.setEnabled(enabled)

    def set_device_info(self, info: str) -> None:
        self._device_label.setText(f"Device: {info}")

    def set_status(self, status: str) -> None:
        self._status_label.setText(status)

    def _on_model_changed(self) -> None:
        model_id = self._model_combo.currentData()
        if model_id and model_id in MODEL_REGISTRY:
            entry = MODEL_REGISTRY[model_id]
            self._model_desc.setText(
                f"{entry.description}\nScale: {entry.scale}x | Size: {entry.file_size_mb:.0f} MB"
            )
            self.model_changed.emit(model_id)
