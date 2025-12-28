from PyQt5.QtWidgets import (
    QDialog, QSlider, QFormLayout, QDialogButtonBox,
    QHBoxLayout, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg
import numpy as np
import midi

class OptionsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.parent = parent
        self.scale_factor = 10  # Use 10 to map 0.1 steps to integer slider values

        layout = QFormLayout()

        self.controls = {}

        for i in midi.settings:
            self.slider, self.spin = self.create_slider_spin(
                i, midi.settings[i][0], self.update_settings(midi.settings, i), 0.0, midi.settings[i][2], 0.01
            )
            layout.addRow(i, self.wrap_in_hbox(self.slider, self.spin))


        # --- Spectrum ---
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setYRange(0, 10)
        layout.addRow(self.plotWidget)


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(20)

        # --- Close button ---
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def update_spectrum(self):
        x = np.arange(0, len(self.parent.smoothed_spectrum))
        self.plotWidget.clear()
        self.plotWidget.plot(x, np.log(self.parent.smoothed_spectrum + 1))

        if midi.settings_updated["update"]:
            for i in midi.settings:
                self.controls[i]["spinbox"].setValue(midi.settings[i][0])
            midi.settings_updated["update"] = False
            print("Yes")



    # Helper to create slider and spinbox pair
    def create_slider_spin(self, name, value, slot, min_val, max_val, step=0.1):
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

        self.controls[name] = {
            "slider": slider,
            "spinbox": spinbox,
        }

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


    def update_tiling_from_spin(self, value):
        self.parent.tiling = value


    def update_settings(self, settings, name):
        def update(value):
            settings[name][0] = value
        return update
