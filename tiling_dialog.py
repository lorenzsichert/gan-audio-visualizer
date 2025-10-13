from PyQt5.QtWidgets import (
    QDialog, QSlider, QFormLayout, QDialogButtonBox,
)

from PyQt5.QtCore import Qt


class OptionsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.parent = parent

        layout = QFormLayout()
        # Audio Weight Slider
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(int(self.parent.audio_weight * 100))
        self.noise_slider.valueChanged.connect(self.update_audio)
        layout.addRow("Audio Weight:", self.noise_slider)

        # Noise Weight Slider
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(int(self.parent.noise_weight * 100))
        self.noise_slider.valueChanged.connect(self.update_noise)
        layout.addRow("Noise Weight:", self.noise_slider)

        # Smoothing Factor Slider
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(100)
        self.smoothing_slider.setValue(int(self.parent.smoothing_factor * 100))
        self.smoothing_slider.valueChanged.connect(self.update_smoothing)
        layout.addRow("Smoothing Factor:", self.smoothing_slider)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def update_audio(self, value):
        self.parent.audio_weight = value / 100.0

    def update_noise(self, value):
        self.parent.noise_weight = value / 100.0

    def update_smoothing(self, value):
        self.parent.smoothing_factor = value / 100.0
