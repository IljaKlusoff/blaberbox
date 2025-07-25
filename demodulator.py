import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SD_NO_ALSA_WARNINGS"] = "1"

import sounddevice as sd
import numpy as np
import queue
import threading
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
import opuslib
from scipy.fft import rfft, rfftfreq

class VoiceDecoder:
    def __init__(
        self,
        sample_rate=16000,
        chirp_duration=0.01,  # 10ms
        frame_duration=0.02,  # Opus expects 20ms
        key=None,
        nonce=None,
        input_device=None
    ):
        self.SAMPLE_RATE = sample_rate
        self.CHIRP_DURATION = chirp_duration
        self.CHIRP_SAMPLES = int(self.SAMPLE_RATE * self.CHIRP_DURATION)
        self.FRAME_SAMPLES = int(self.SAMPLE_RATE * frame_duration)

        self.CHIRP_BASE = 200
        self.CHIRP_STEP = 15

        self.KEY = key or os.urandom(32)
        self.NONCE = nonce or os.urandom(16)
        cipher = Cipher(algorithms.ChaCha20(self.KEY, self.NONCE), mode=None)
        self.decryptor = cipher.decryptor()

        self.decoder = opuslib.Decoder(self.SAMPLE_RATE, 1)
        self.byte_buffer = bytearray()
        self.audio_buffer = queue.Queue()
        self._running = False
        self._stream = None
        self._thread = None
        self.input_device = input_device

    def _detect_byte_from_chirp(self, chirp_samples):
        spectrum = np.abs(rfft(chirp_samples * np.hamming(len(chirp_samples))))
        freqs = rfftfreq(len(chirp_samples), 1 / self.SAMPLE_RATE)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        byte = round((peak_freq - self.CHIRP_BASE) / self.CHIRP_STEP)
        if 0 <= byte <= 255:
            return byte
        return None

    def _demodulate_worker(self):
        frame_buf = []
        while self._running:
            try:
                chirp = self.audio_buffer.get(timeout=0.5)
            except queue.Empty:
                continue
            byte = self._detect_byte_from_chirp(chirp)
            if byte is not None:
                self.byte_buffer.append(byte)

            if len(self.byte_buffer) >= 10:  # adjust as needed for Opus frames
                try:
                    decrypted = self.decryptor.update(bytes(self.byte_buffer[:10]))
                    pcm = self.decoder.decode(decrypted, self.FRAME_SAMPLES)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
                    sd.play(audio, self.SAMPLE_RATE)
                except Exception as e:
                    pass
                del self.byte_buffer[:10]

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print("⚠️", status)
        mono = indata[:, 0]
        for i in range(0, len(mono), self.CHIRP_SAMPLES):
            chunk = mono[i:i+self.CHIRP_SAMPLES]
            if len(chunk) == self.CHIRP_SAMPLES:
                self.audio_buffer.put(chunk.copy())

    def start(self):
        if self._running:
            return
        print("🔓 VoiceDecoder started — listening for chirps...")
        self._running = True
        self._thread = threading.Thread(target=self._demodulate_worker, daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHIRP_SAMPLES,
            channels=1,
            dtype='float32',
            latency="low",
            callback=self._audio_callback,
            device=self.input_device
        )
        self._stream.start()

    def stop(self):
        if not self._running:
            return
        print("🛑 VoiceDecoder stopping...")
        self._running = False
        self._thread.join()
        self._stream.stop()
        self._stream.close()
        print("✅ Decoder stopped.")

if __name__ == "__main__":
    # Use the same key/nonce as your modulator
    KEY = b'\x01\x02\x03\x04' * 8  # 32 bytes
    NONCE = b'\x05\x06\x07\x08' * 4  # 16 bytes
    decoder = VoiceDecoder(key=KEY, nonce=NONCE)
    try:
        decoder.start()
        print("🎧 Listening for encrypted audio. Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        decoder.stop()
