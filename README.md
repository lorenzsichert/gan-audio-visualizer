# Audio GAN Visualizer (Qt)

A real-time **audio-driven GAN visualizer** built with PyQt5 and PyTorch. This application generates dynamic images using a pre-trained GAN, influenced by live audio input. Great for exploring generative visuals controlled by sound.  

![Example Screenshot](gif/cifar10.gif)


## Features

- Real-time audio input visualization.
- GAN-based image generation using pre-trained models.
- Adjustable latent vector dynamics and audio influence.
- Model and input device switching


## Installation

1. **Clone the repository**

```bash
git clone https://github.com/lorenzsichert/audio-gan-visualizer.git
cd audio-gan-visualizer
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

- Python ≥ 3.10
- PyQt5
- NumPy
- PyTorch
- Torchvision
- sounddevice
- scipy

Install via:

```bash
pip install pyqt5 numpy torch torchvision sounddevice scipy
```
Or via requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Ensure you have audio input devices available** (microphone, line-in, etc.)
## Usgae
```
python main.py
```


## Example Screenshots

---

## Code Overview

- `main.py` → Main application script with GUI, audio input, and GAN visualization logic.
- `dcgan.py` → GAN architecture definition (ColorGenerator).
- `models_dialog.py` → Model selection dialog.
- `options_dialog.py` → Slider adjustments dialog.
- `input_dialog.py` → Audio input selection dialog.
- `recording.py` → Functions for sampling and smoothing audio input.

---
