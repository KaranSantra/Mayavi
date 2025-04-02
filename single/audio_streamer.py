import numpy as np
import queue
import threading


class AudioStreamer:
    def __init__(self):
        self.sent_chunks = []
        self.running = False
        self.chunk_queue = queue.Queue()
        self.stream_thread = None

    def start_streaming(self):
        """Start the streaming process."""
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_audio)
        self.stream_thread.start()

    def send_chunk(self, audio_chunk):
        """Add an audio chunk to the queue for streaming."""
        self.sent_chunks.append(audio_chunk)  # Store chunk immediately for testing
        self.chunk_queue.put(audio_chunk)

    def _stream_audio(self):
        """Process and send audio chunks from the queue."""
        while self.running:
            try:
                chunk = self.chunk_queue.get(timeout=0.1)
                # In a real implementation, this would send the chunk to a server
            except queue.Empty:
                continue

    def stop(self):
        """Stop the streaming process."""
        self.running = False
        if self.stream_thread:
            self.stream_thread.join()
