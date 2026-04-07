from functools import partial
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
import mido
from numpy import unsignedinteger

import midi
import stylesheets

class MidiDialog(QDialog):
    def __init__(self, parent, worker):
        super().__init__(parent)
        self.setStyleSheet(stylesheets.stylesheet)
        self.setWindowTitle("MIDI Settings")
        self.setMinimumWidth(800)
        self.parent = parent

        self.header_devices = QLabel("MIDI Device:")


        self.midi_device = self.parent.midi_device

        self.dropdown = QComboBox()
        self.load_devices()
        self.dropdown.currentIndexChanged.connect(self.input_changed)

        index = self.dropdown.findText(self.midi_device)
        print(index)
        if index != -1:
            self.dropdown.setCurrentIndex(index)

        self.apply_button = QDialogButtonBox(QDialogButtonBox.Apply)
        self.apply_button.clicked.connect(self.apply)


        self.header_mapping = QLabel("Input Mapping:")

        self.layout = QFormLayout()
        self.layout.addWidget(self.header_devices)
        self.layout.addWidget(self.dropdown)

        self.layout.addWidget(self.header_mapping)

        self.buttons = {}
        
        for i in midi.settings:
            label = QLabel(f"{i}")
            layout = QHBoxLayout()
            if midi.settings[i][3] != -1:
                button = QPushButton(f"Press to Remap MIDI: {midi.settings[i][3]}")
            else:
                button = QPushButton(f"Press to Remap MIDI: {midi.settings[i][3]}")
            button.clicked.connect(partial(self.remap_midi_control, i, midi.settings[i][3]))
            unset_button = QPushButton("X")
            unset_button.clicked.connect(partial(self.unset_midi_control, i))
            layout.addWidget(label)
            layout.addWidget(button)
            layout.addWidget(unset_button)
            unset_button.setFixedWidth(50)
            pressed = False
            value = midi.settings[i][3]
            self.buttons[i] = [label, button, pressed, value]
            self.layout.addRow(layout)


        self.layout.addWidget(self.apply_button)
        self.setLayout(self.layout)

        worker.finished.connect(self.set_midi_control)

    def remap_midi_control(self, name, value):
        print(f"Set {name} to {value}!")
        self.buttons[name][1].setText("Move MIDI control to set")
        self.buttons[name][2] = True
        midi.settings[name][3] = -2

    def unset_midi_control(self, name):
        self.buttons[name][1].setText("MIDI control not set")
        self.buttons[name][2] = False
        midi.settings[name][3] = -1
        self.buttons[name][3] = -1

    def set_midi_control(self, value):
        for i in self.buttons:
            if self.buttons[i][2] == True:
                self.buttons[i][3] = value
                self.buttons[i][2] = False
                self.buttons[i][1].setText(f"Press to Remap MIDI: {midi.settings[i][3]}")

        print(value)

    def input_changed(self):
        self.midi_device = self.dropdown.currentText()
        print(self.midi_device)



    def load_devices(self):
        input_devices = mido.get_input_names()
        for i in input_devices:
            self.dropdown.addItem(i)

    def apply(self):
        self.parent.midi_device = self.midi_device
        print(self.midi_device)
        self.parent.open_midi_device()
        self.parent.settings.setValue("midi_device", self.parent.midi_device)

    def reject(self) -> None:
        open_button = False
        for i in self.buttons:
            if self.buttons[i][2] == True:
                open_button = True
                self.buttons[i][2] = False
                self.buttons[i][1].setText(f"Press to Remap MIDI: {self.buttons[i][3]}")

        
        if open_button == False:
            return super().reject()


