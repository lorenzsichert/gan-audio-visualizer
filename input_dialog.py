from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel
import sounddevice as sd


class InputDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent) 
        self.setWindowTitle("Input Options")
        self.parent = parent

        layout = QFormLayout()


        self.dropdown = QComboBox()
        self.load_devices()
        self.dropdown.currentIndexChanged.connect(self.input_changed)

        self.label = QLabel()
        self.selected_device = None

        if self.parent.device == None:
            self.label = QLabel("No device selected. Choose an Input Source:")
        else:
            self.selected_device = self.parent.device
            self.update_text()


        self.apply_button = QDialogButtonBox(QDialogButtonBox.Apply)
        self.apply_button.clicked.connect(self.apply)

        layout.addWidget(self.label)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def load_devices(self):
        self.dropdown.clear()
        for i in sd.query_devices():
            self.dropdown.addItem(i['name'], {"name": i['name'], "index": i['index'], "default_samplerate": i['default_samplerate'], "max_input_channels": i["max_input_channels"]})


    def input_changed(self):
        self.selected_device = self.dropdown.currentData()
        self.update_text()
    
    def update_text(self):
        self.label.setText(f"Selected Device: {self.selected_device["name"]}, Sample Rate: {self.selected_device["default_samplerate"]}")

    def apply(self):
        self.parent.device = self.dropdown.currentData()
        self.parent.settings.setValue("input_device", self.parent.device)
        self.parent.open_stream()

