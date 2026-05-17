from PyQt5.QtCore import QObject, pyqtSignal
import mido

# [Value, Min, Max, MIDI id, 0: knob - 1: spin wheel]
settings = {
    "Smoothing Factor": [0.16, 0, 1, 63, 0],
    "Noise Weight": [1.48, 0, 10, 18, 0],
    "Audio Weight": [0.1, 0, 2.0, 17, 0],
    "Audio Randomization": [0.0, 0, 1, -1, 1],
    "Lowpass Sensivity": [0.47, 0, 5.0, 122, 0],
    "Lowpass Power": [1.39, 1.0, 2.0, 57, 0],
    "Lowpass Cutoff": [128, 0, 256, 59, 0],
    "Noise Injection": [0, 0, 5, 61, 0],
    "Noise Base": [1.0, 0, 20, -1, 1],
    "Hue Shift": [0.09, 0, 1.0, -1, 1],
    "Y": [0.07, 0, 1, -1, 1],
    "X": [0.17, -0.5, 1, -1, 1],
    "Cutoff": [0, 0, 100, -1, 0],
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
                        if msg.value == 127 or msg.value == 126 or msg.value == 1 or msg.value == 2:
                            settings[i][3] = msg.control
                            settings[i][4] = 1
                        else:
                            settings[i][3] = msg.control
                            settings[i][4] = 0
                        self.finished.emit(msg.control)
                    elif msg.control == settings[i][3]:
                        if settings[i][4] == 1:
                            step = settings[i][2] / 256.0
                            if msg.value == 127 or msg.value == 126:
                                settings[i][0] -= step
                            else:
                                settings[i][0] += step
                            if settings[i][0] > settings[i][2]:
                                settings[i][0] = settings[i][2]
                            if settings[i][0] < settings[i][1]:
                                settings[i][0] = settings[i][1]
                        else:
                            settings[i][0] = msg.value / (128.0 / settings[i][2]) + settings[i][1]

        return callback




