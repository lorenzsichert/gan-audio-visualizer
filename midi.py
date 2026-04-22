from PyQt5.QtCore import QObject, pyqtSignal
import mido

settings = {
    "Smoothing Factor": [0.5, 0, 1, 63],
    "Noise Weight": [1.0, 0, 10, 18],
    "Audio Weight": [0.1, 0, 0.5, 17],
    "Audio Randomization": [0.5, 0, 1, -1],
    "Lowpass Sensivity": [10, 0, 20, 122],
    "Lowpass Power": [1.0, 1.0, 2.0, 57],
    "Lowpass Cutoff": [30, 0, 256, 59],
    "Noise Injection": [1, 0, 5, 61],
    "Noise Base": [0.7, 0, 20, -1],
    "Hue Shift": [10, 0, 50, -1],
    "Red": [0, 0, 1, -1],
    "Green": [0, 0, 1, -1],
    "Blue": [0, 0, 1, -1],
}

settings_updated = {
    "update": False
}


class Worker(QObject):
    finished = pyqtSignal(int)


    def open_midi_device(self, midi_device, settings):
        port_name = midi_device

        try:
            midi = mido.open_input(port_name, callback=self.midi_callback(settings))
            print(f"Succesfully opened MIDI Device: {port_name}")
        except Exception as e:
            print(f"Error opening {port_name}: {e}")
            midi = None

        return midi
        
    def midi_callback(self, settings):
        def callback(msg):
            if msg.type == "control_change":
                settings_updated["update"] = True
                for i in settings:
                    if settings[i][3] == -2:
                        settings[i][3] = msg.control
                        self.finished.emit(msg.control)
                    if msg.control == settings[i][3]:
                        settings[i][0] = msg.value / (128.0 / settings[i][2]) + settings[i][1]

        return callback
