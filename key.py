import sounddevice as sd
import numpy as np
import librosa
import queue
import threading
import time
from scipy.stats import pearsonr

# Audio stream settings
SAMPLERATE = 22050
BLOCKSIZE = 1024
BUFFER_SECONDS = 5
DETECTION_INTERVAL = 5  # How often to estimate key

# Rolling audio buffer
audio_buffer = queue.Queue()
rolling_audio = np.zeros(SAMPLERATE * BUFFER_SECONDS)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    # Flatten stereo to mono
    mono = np.mean(indata, axis=1)
    audio_buffer.put(mono)

def key_from_audio(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    keys = []
    scores = []

    for i in range(12):
        major = np.roll(major_profile, i)
        minor = np.roll(minor_profile, i)
        keys += [librosa.midi_to_note(i + 60) + ' major',
                 librosa.midi_to_note(i + 60) + ' minor']
        scores += [pearsonr(chroma_avg, major)[0],
                   pearsonr(chroma_avg, minor)[0]]

    best_idx = np.argmax(scores)
    return keys[best_idx]

def process_audio():
    global rolling_audio
    while True:
        # Collect enough audio for BUFFER_SECONDS
        while not audio_buffer.empty():
            data = audio_buffer.get()
            rolling_audio = np.concatenate((rolling_audio[len(data):], data))

        # Once every DETECTION_INTERVAL seconds, estimate key
        key = key_from_audio(rolling_audio, SAMPLERATE)
        print("Detected key:", key)

        time.sleep(DETECTION_INTERVAL)

# Start processing thread
threading.Thread(target=process_audio, daemon=True).start()

# Start audio stream
with sd.InputStream(channels=1, callback=audio_callback,
                    blocksize=BLOCKSIZE, samplerate=SAMPLERATE):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
