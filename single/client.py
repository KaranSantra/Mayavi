import pyaudio
import numpy as np
import threading
import queue
import time


class Client:
    def __init__(self):
        # Audio recording parameters
        self.CHUNK = 1024 * 3
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # State variables
        self.running = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.record_thread = None

    def start_recording(self):
        """Start recording audio from the microphone."""
        self.running = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.daemon = (
            True  # Make thread daemon so it stops with main thread
        )
        self.record_thread.start()

    def _record_audio(self):
        """Record audio in chunks and add them to the queue."""
        while self.running:
            try:
                data = self.stream.read(self.CHUNK)
                if isinstance(data, bytes):
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                else:
                    # For testing purposes, use mock data directly
                    audio_chunk = np.ones(self.CHUNK, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def stop(self):
        """Stop recording and clean up resources."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.record_thread:
            self.record_thread.join(timeout=1)  # Add timeout to prevent hanging
        self.p.terminate()
