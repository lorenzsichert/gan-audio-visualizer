import mido

settings = {
    "Smoothing Factor": [0.5, 0, 1, 63],
    "Noise Weight": [1.0, 0, 2, 60],
    "Audio Weight": [0.1, 0, 2, 64],
    "Audio Randomization": [0.5, 0, 1, -1],
    "Lowpass Sensivity": [10, 0, 25, 54],
    "Lowpass Power": [1.0, 1.0, 2.0, 57],
    "Lowpass Cutoff": [30, 0, 256, 59],
}

settings_updated = {
    "update": False
}

def open_midi_device(settings):
    port_name = "DJControl Compact"

    try:
        midi = mido.open_input(port_name, callback=midi_callback(settings))
    except Exception as e:
        print(f"Error opening {port_name}: {e}")
        midi = None

    return midi
    
def midi_callback(settings):
    def callback(msg):
        if msg.type == "control_change":
            settings_updated["update"] = True
            print(msg)
            for i in settings:
                if msg.control == settings[i][3]:
                    settings[i][0] = msg.value / (128.0 / settings[i][2]) + settings[i][1]

    return callback
