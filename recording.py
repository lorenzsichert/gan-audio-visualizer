import numpy as np
from scipy.fft import rfft
import sounddevice as sd
import torch

def open_stream(device, blocksize):
    if device == None:
        stream = None
        return None
    
    if device['max_input_channels'] < 1:
        print(f"❌ {device['name']} has no input channels.")
        return None

    try:
        print(sd.check_input_settings( 
            samplerate=device['default_samplerate'],
            channels=device['max_input_channels'],
            dtype='float32',
            device=device['name']
        ))

        stream = sd.InputStream(
            samplerate=device['default_samplerate'],
            blocksize=blocksize,
            channels=device['max_input_channels'],
            dtype='float32',
            device=device['name']
        )
        stream.start()
        print(f"✅ Stream started succesfully on device: {device}!")
    except sd.PortAudioError as e:
        print(f"❌ Failed to start Stream: {e}")
        stream = None
        device = None
    except Exception as e:
        print(f"❌ Failed to start Stream: {e}")
        stream = None
        device = None
    return stream

def get_sample(stream, smoothed_spectrum, blocksize, smoothing_factor):
    # Audio Stream Input
    recording, _ = stream.read(blocksize)

    samples = recording[:,0]
    window = np.hanning(len(samples)) * samples
    fft_spectrum = np.abs(rfft(window))

    smoothed_spectrum = (
        smoothing_factor * smoothed_spectrum +
        (1 - smoothing_factor) * fft_spectrum
    )
    return fft_spectrum

def push_latent(z, direction, epsilon):
    z_pushed = z + (direction * epsilon)

    z_final = z_pushed * (z.norm()/z_pushed.norm())

    return z_final
