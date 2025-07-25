import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SD_NO_ALSA_WARNINGS"] = "1"

import sounddevice as sd
import numpy as np
import queue
import threading
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
import opuslib
from scipy.signal import chirp

# === CONFIGURATION ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.02  # 20 ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION)
CHIRP_DURATION = 0.01  # 10 ms chirps (faster)
CHIRP_SAMPLES = int(SAMPLE_RATE * CHIRP_DURATION)
CHIRP_START_FREQ = 1000
CHIRP_END_FREQ = 4000
THRESHOLD = 0.01

# === ENCRYPTION SETUP ===
KEY = os.urandom(32)
NONCE = os.urandom(16)
cipher = Cipher(algorithms.ChaCha20(KEY, NONCE), mode=None)
encryptor = cipher.encryptor()

# === OPUS ENCODER SETUP ===
opus_encoder = opuslib.Encoder(SAMPLE_RATE, 1, opuslib.APPLICATION_AUDIO)
opus_encoder.bitrate = 6000  # Reduce bitrate for fewer bytes per frame

# === AUDIO BUFFERS ===
audio_queue = queue.Queue()
output_buffer = queue.Queue()

# === CHIRP ENCODING ===
def generate_chirp(freq_offset):
    t = np.linspace(0, CHIRP_DURATION, CHIRP_SAMPLES, endpoint=False)
    base = chirp(t, f0=CHIRP_START_FREQ, f1=CHIRP_END_FREQ, t1=CHIRP_DURATION, method='linear')
    shift = np.sin(2 * np.pi * freq_offset * t)
    return (base * shift).astype(np.float32)

def byte_to_chirp(byte_val):
    # Wider spacing to avoid tone collisions
    freq_offset = 200 + (byte_val * 15)
    return generate_chirp(freq_offset)

# === WORKER THREAD ===
def modulate_worker():
    while True:
        frame = audio_queue.get()
        if frame is None:
            break
        pcm = np.int16(frame * 32767)
        encoded = opus_encoder.encode(pcm.tobytes(), FRAME_SAMPLES)
        encrypted = encryptor.update(encoded)
        for byte in encrypted:
            output_buffer.put(byte_to_chirp(byte))

# === AUDIO CALLBACK ===
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)
    mono = indata[:, 0]
    if np.abs(mono).mean() > THRESHOLD:
        audio_queue.put(mono.copy())
    out = np.zeros(frames, dtype=np.float32)
    i = 0
    while i < frames:
        try:
            tone = output_buffer.get_nowait()
            n = min(len(tone), frames - i)
            out[i:i+n] = tone[:n]
            if n < len(tone):
                output_buffer.queue.appendleft(tone[n:])
            i += n
        except queue.Empty:
            break
    outdata[:, 0] = out

# === MAIN LOOP ===
def main():
    print("🚀 Encrypted voice-over-chirp modulator (10ms chirps)")
    worker = threading.Thread(target=modulate_worker, daemon=True)
    worker.start()
    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=1024,
            channels=1,
            dtype='float32',
            latency=(0.1, 0.1),
            callback=audio_callback
        ):
            print("🎙️ Speak into the mic. Fast encrypted chirps will play. Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        audio_queue.put(None)
        worker.join()

if __name__ == "__main__":
    main()
