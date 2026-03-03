import mido

settings = {
    "Smoothing Factor": [0.5, 0, 1, 63],
    "Noise Weight": [1.0, 0, 10, 18],
    "Audio Weight": [0.1, 0, 2, 17],
    "Audio Randomization": [0.5, 0, 1, -1],
    "Lowpass Sensivity": [10, 0, 20, 122],
    "Lowpass Power": [1.0, 1.0, 2.0, 57],
    "Lowpass Cutoff": [30, 0, 256, 59],
    "Noise Injection": [1, 0, 5, 61],
    "Noise Base": [0.7, 0, 20, -1],
    "Hue Shift": [10, 0, 50, -1],
}

settings_updated = {
    "update": False
}

def open_midi_device(settings):
    print("Available output ports:", mido.get_input_names())

    port_name = "XDJ-RX3:XDJ-RX3 MIDI 1 28:0"

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
