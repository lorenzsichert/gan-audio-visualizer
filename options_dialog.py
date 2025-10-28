from PyQt5.QtWidgets import (
    QDialog, QSlider, QFormLayout, QDialogButtonBox,
    QHBoxLayout, QDoubleSpinBox
)
from PyQt5.QtCore import Qt


class OptionsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.parent = parent
        self.scale_factor = 10  # Use 10 to map 0.1 steps to integer slider values

        layout = QFormLayout()

        # --- Audio Weight ---
        self.audio_slider, self.audio_spin = self.create_slider_spin(
            self.parent.audio_weight, self.update_audio, 0.0, 1.0, 0.01
        )
        layout.addRow("Audio Weight:", self.wrap_in_hbox(self.audio_slider, self.audio_spin))

        # --- Noise Weight ---
        self.noise_slider, self.noise_spin = self.create_slider_spin(
            self.parent.noise_weight, self.update_noise, 0.0, 1.0, 0.01
        )
        layout.addRow("Noise Weight:", self.wrap_in_hbox(self.noise_slider, self.noise_spin))

        # --- Smoothing Factor ---
        self.smoothing_slider, self.smoothing_spin = self.create_slider_spin(
            self.parent.smoothing_factor, self.update_smoothing, 0.0, 1.0, 0.01
        )
        layout.addRow("Smoothing Factor:", self.wrap_in_hbox(self.smoothing_slider, self.smoothing_spin))

        # --- Audio Randomization Factor ---
        self.audio_r_slider, self.audio_r_spin = self.create_slider_spin(
            self.parent.audio_randomization, self.update_audio_r, 0.0, 30.0, 0.01
        )
        layout.addRow("Audio Randomization:", self.wrap_in_hbox(self.audio_r_slider, self.audio_r_spin))

        # --- Noise Randomization Factor ---
        self.noise_r_slider, self.noise_r_spin = self.create_slider_spin(
            self.parent.noise_randomization, self.update_noise_r, 0.0, 30.0, 0.01
        )
        layout.addRow("Noise Randomization:", self.wrap_in_hbox(self.noise_r_slider, self.noise_r_spin))

        # --- Tiling ---
        self.h_slider, self.h_spinbox = self.create_slider_spin(
            self.parent.tiling, self.update_tiling_from_spin, 1.0, 10.0, 1.0
        )
        layout.addRow("Tiling:", self.wrap_in_hbox(self.h_slider, self.h_spinbox))

        # --- Close button ---
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    # Helper to create slider and spinbox pair
    def create_slider_spin(self, value, slot, min_val, max_val, step=0.1):
        factor = int(1 / step)  # maps step to integer slider values
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * factor))
        slider.setMaximum(int(max_val * factor))
        slider.setValue(int(value * factor))

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(value)

        slider.valueChanged.connect(lambda v: self.sync_slider_spin(spinbox, v, factor, slot))
        spinbox.valueChanged.connect(lambda v: self.sync_spin_slider(slider, v, factor, slot))

        return slider, spinbox

    # Helper to wrap slider and spinbox in a horizontal layout
    def wrap_in_hbox(self, slider, spinbox):
        h_layout = QHBoxLayout()
        h_layout.addWidget(slider)
        h_layout.addWidget(spinbox)
        return h_layout

    # Sync methods
    def sync_slider_spin(self, spinbox, value, factor, slot):
        spinbox.blockSignals(True)
        spinbox.setValue(value / factor)
        spinbox.blockSignals(False)
        slot(value / factor)

    def sync_spin_slider(self, slider, value, factor, slot):
        slider.blockSignals(True)
        slider.setValue(int(value * factor))
        slider.blockSignals(False)
        slot(value)

    # Update methods
    def update_audio(self, value):
        self.parent.audio_weight = value

    def update_noise(self, value):
        self.parent.noise_weight = value

    def update_smoothing(self, value):
        self.parent.smoothing_factor = value

    def update_tiling_from_spin(self, value):
        self.parent.tiling = value

    def update_audio_r(self, value):
        self.parent.audio_randomization = value

    def update_noise_r(self, value):
        self.parent.noise_randomization = value
