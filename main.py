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
import sounddevice as sd


class VoiceModulator:
    def __init__(
        self,
        sample_rate=16000,
        chirp_duration=0.01,  # 10 ms
        frame_duration=0.02,  # 20 ms
        opus_bitrate=6000,
        threshold=0.01,
        key=None,
        nonce=None,
    ):
        self.SAMPLE_RATE = sample_rate
        self.CHIRP_DURATION = chirp_duration
        self.FRAME_DURATION = frame_duration
        self.FRAME_SAMPLES = int(self.SAMPLE_RATE * self.FRAME_DURATION)
        self.CHIRP_SAMPLES = int(self.SAMPLE_RATE * self.CHIRP_DURATION)
        self.THRESHOLD = threshold

        self.CHIRP_START_FREQ = 1000
        self.CHIRP_END_FREQ = 4000

        self.KEY = key or os.urandom(32)
        self.NONCE = nonce or os.urandom(16)
        cipher = Cipher(algorithms.ChaCha20(self.KEY, self.NONCE), mode=None)
        self.encryptor = cipher.encryptor()

        self.encoder = opuslib.Encoder(self.SAMPLE_RATE, 1, opuslib.APPLICATION_AUDIO)
        self.encoder.bitrate = opus_bitrate

        self.audio_queue = queue.Queue()
        self.output_buffer = queue.Queue()

        self._stream = None
        self._worker_thread = None
        self._running = False

    def _generate_chirp(self, freq_offset):
        t = np.linspace(0, self.CHIRP_DURATION, self.CHIRP_SAMPLES, endpoint=False)
        base = chirp(t, f0=self.CHIRP_START_FREQ, f1=self.CHIRP_END_FREQ, t1=self.CHIRP_DURATION, method='linear')
        shift = np.sin(2 * np.pi * freq_offset * t)
        return (base * shift).astype(np.float32)

    def _byte_to_chirp(self, byte_val):
        freq_offset = 200 + (byte_val * 15)
        return self._generate_chirp(freq_offset)

    def _modulate_worker(self):
        while self._running:
            frame = self.audio_queue.get()
            if frame is None:
                break
            pcm = np.int16(frame * 32767)
            encoded = self.encoder.encode(pcm.tobytes(), self.FRAME_SAMPLES)
            encrypted = self.encryptor.update(encoded)
            for byte in encrypted:
                self.output_buffer.put(self._byte_to_chirp(byte))

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print("⚠️", status)
        mono = indata[:, 0]
        if np.abs(mono).mean() > self.THRESHOLD:
            self.audio_queue.put(mono.copy())
        out = np.zeros(frames, dtype=np.float32)
        i = 0
        while i < frames:
            try:
                tone = self.output_buffer.get_nowait()
                n = min(len(tone), frames - i)
                out[i:i+n] = tone[:n]
                if n < len(tone):
                    self.output_buffer.queue.appendleft(tone[n:])
                i += n
            except queue.Empty:
                break
        outdata[:, 0] = out

    def start(self):
        if self._running:
            return
        print("🔐 VoiceModulator started.")
        print(sd.query_devices())
        self._running = True
        self._worker_thread = threading.Thread(target=self._modulate_worker, daemon=True)
        self._worker_thread.start()
        self._stream = sd.Stream(
            samplerate=self.SAMPLE_RATE,
            blocksize=1024,
            channels=1,
            dtype='float32',
            latency=(0.1, 0.1),
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self):
        if not self._running:
            return
        print("🛑 VoiceModulator stopping...")
        self._running = False
        self.audio_queue.put(None)
        self._worker_thread.join()
        
        self._stream.stop()
        self._stream.close()
        print("✅ Stopped.")

if __name__ == "__main__":
    key = b'\x01\x02\x03\x04' * 8
    nonce = b'\x05\x06\x07\x08' * 4
    vm = VoiceModulator(key=key, nonce=nonce)
    vm.start()
    try:
        print("🎙️ Speak into the mic — encrypted chirps playing. Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        vm.stop()
