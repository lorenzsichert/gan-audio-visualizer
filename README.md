# Audio GAN Visualizer (Qt)

A real-time **audio-driven GAN visualizer** built with PyQt5 and PyTorch. Captures live audio input (microphone, line-in, or MIDI) and uses it to modulate a latent vector feeding a pre-trained GAN, producing dynamic visuals that react to sound.


## Requirements

- **Python 3.14** (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lorenzsichert/gan-audio-visualizer.git
cd gan-audio-visualizer
```

### 2. Create a virtual environment (recommended)

```bash
python3.14 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision sounddevice pyaudio pyqt5 scipy mido python-rtmidi pyqtgraph
```

> **Note:** PyTorch installation varies by platform and CUDA version. See [pytorch.org](https://pytorch.org) for platform-specific instructions.



### 4. Ensure audio input is available

A working microphone, line-in, or virtual audio cable is required.

On Linux: Native playback recording is possible

## Usage

```bash
python main.py
```

Use the **Options** menu to:
- **Adjust Sliders** — tweak smoothing, weights, randomization, hue shift, etc.
- **Model Settings** — switch between SLE conditional, SLE upscale, or ONNX models
- **Input Settings** — select audio input device
- **MIDI Settings** — configure MIDI controller bindings
- **Open Fullscreen Window** — display output in a separate fullscreen window

Press **Esc** to quit.

## Code Overview

| File | Description |
|---|---|
| `main.py` | Main application window, audio loop, GAN inference, rendering |
| `models/conditional/sle_conditional.py` | Conditional SLE generator with noise injection |
| `models/upscale/sle.py` | Upscale SLE generator |
| `models_dialog.py` | Model selection / loading dialog |
| `options_dialog.py` | Parameter slider adjustments dialog |
| `input_dialog.py` | Audio input device selection dialog |
| `midi.py` | MIDI controller worker |
| `midi_dialog.py` | MIDI settings dialog |
| `recording.py` | Audio sampling and latent vector smoothing utilities |
| `fullscreen.py` | Fullscreen display window |
| `stylesheets.py` | Qt stylesheet definitions |

## License

MIT
