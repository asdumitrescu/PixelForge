"""Settings dialog for user preferences."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.constants import MIN_TILE_SIZE
from src.engine.settings import AppSettings


class SettingsDialog(QDialog):
    """Application settings dialog with Inference, Output, and About tabs."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 380)

        layout = QVBoxLayout(self)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_inference_tab(), "Inference")
        tabs.addTab(self._create_output_tab(), "Output")
        layout.addWidget(tabs)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_inference_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tile size
        tile_group = QGroupBox("Tile Size")
        tile_layout = QVBoxLayout(tile_group)

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Size:"))
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(MIN_TILE_SIZE, 512)
        self._tile_spin.setSingleStep(32)
        self._tile_spin.setValue(self._settings.tile_size)
        self._tile_spin.setSuffix(" px")
        tile_row.addWidget(self._tile_spin)
        tile_layout.addLayout(tile_row)

        tile_hint = QLabel(
            "Lower = less VRAM, slower. 256 is safe for 4GB GPUs."
        )
        tile_hint.setStyleSheet("color: #a6adc8; font-size: 11px;")
        tile_hint.setWordWrap(True)
        tile_layout.addWidget(tile_hint)

        layout.addWidget(tile_group)

        # Precision
        precision_group = QGroupBox("Precision")
        precision_layout = QVBoxLayout(precision_group)
        self._half_check = QCheckBox("Use half precision (fp16)")
        self._half_check.setChecked(self._settings.use_half)
        precision_layout.addWidget(self._half_check)
        precision_hint = QLabel("Saves ~50% VRAM. No speed benefit on GTX 10xx.")
        precision_hint.setStyleSheet("color: #a6adc8; font-size: 11px;")
        precision_layout.addWidget(precision_hint)
        layout.addWidget(precision_group)

        # Default model
        model_group = QGroupBox("Default Model")
        model_layout = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        from src.engine.model_registry import MODEL_REGISTRY

        for model_id, entry in MODEL_REGISTRY.items():
            self._model_combo.addItem(entry.display_name, userData=model_id)
        # Set current
        idx = self._model_combo.findData(self._settings.default_model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        model_layout.addWidget(self._model_combo)
        layout.addWidget(model_group)

        layout.addStretch()
        return tab

    def _create_output_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Default format
        fmt_group = QGroupBox("Default Output Format")
        fmt_layout = QVBoxLayout(fmt_group)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["PNG", "JPEG", "WebP"])
        idx = self._format_combo.findText(self._settings.output_format)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)
        fmt_layout.addWidget(self._format_combo)
        layout.addWidget(fmt_group)

        # JPEG quality
        jpeg_group = QGroupBox("JPEG Quality")
        jpeg_layout = QHBoxLayout(jpeg_group)
        self._jpeg_slider = QSlider(Qt.Orientation.Horizontal)
        self._jpeg_slider.setRange(50, 100)
        self._jpeg_slider.setValue(self._settings.jpeg_quality)
        self._jpeg_label = QLabel(f"{self._settings.jpeg_quality}%")
        self._jpeg_slider.valueChanged.connect(
            lambda v: self._jpeg_label.setText(f"{v}%")
        )
        jpeg_layout.addWidget(self._jpeg_slider)
        jpeg_layout.addWidget(self._jpeg_label)
        layout.addWidget(jpeg_group)

        # WebP quality
        webp_group = QGroupBox("WebP Quality")
        webp_layout = QHBoxLayout(webp_group)
        self._webp_slider = QSlider(Qt.Orientation.Horizontal)
        self._webp_slider.setRange(50, 100)
        self._webp_slider.setValue(self._settings.webp_quality)
        self._webp_label = QLabel(f"{self._settings.webp_quality}%")
        self._webp_slider.valueChanged.connect(
            lambda v: self._webp_label.setText(f"{v}%")
        )
        webp_layout.addWidget(self._webp_slider)
        webp_layout.addWidget(self._webp_label)
        layout.addWidget(webp_group)

        layout.addStretch()
        return tab

    def _save_and_accept(self) -> None:
        self._settings.tile_size = self._tile_spin.value()
        self._settings.use_half = self._half_check.isChecked()
        self._settings.default_model = self._model_combo.currentData()
        self._settings.output_format = self._format_combo.currentText()
        self._settings.jpeg_quality = self._jpeg_slider.value()
        self._settings.webp_quality = self._webp_slider.value()
        self.accept()
