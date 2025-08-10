import numpy as np
from scipy.fft import rfft

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
    return smoothed_spectrum
