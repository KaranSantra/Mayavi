import numpy as np
import queue
import threading
import socket
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioStreamer:
    def __init__(self, host="localhost", port=5000):
        """Initialize the audio streamer."""
        self.host = host
        self.port = port
        self.sent_chunks = []
        self.running = False
        self.chunk_queue = queue.Queue()
        self.stream_thread = None
        self.socket = None
        self.connected = False
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.logger = logging.getLogger(__name__)

    def start_streaming(self):
        """Start the streaming process."""
        self.running = True
        self.connect()
        self.stream_thread = threading.Thread(target=self._stream_audio)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def connect(self):
        """Establish connection to the server."""
        retries = 0
        while retries < self.max_retries and not self.connected:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                self.logger.info(f"Connected to server at {self.host}:{self.port}")
            except socket.error as e:
                retries += 1
                self.logger.error(f"Connection attempt {retries} failed: {e}")
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def send_chunk(self, audio_chunk):
        """Add an audio chunk to the queue for streaming."""
        self.sent_chunks.append(audio_chunk)  # Store chunk for testing
        self.chunk_queue.put(audio_chunk)

    def _stream_audio(self):
        """Process and send audio chunks from the queue."""
        while self.running:
            try:
                chunk = self.chunk_queue.get(timeout=0.1)
                if not self.connected:
                    self.connect()
                self.socket.send(chunk.tobytes())
            except queue.Empty:
                continue
            except socket.error as e:
                self.logger.error(f"Error sending audio chunk: {e}")
                self.connected = False
                try:
                    self.connect()
                except:
                    break
            except Exception as e:
                self.logger.error(f"Unexpected error in streaming: {e}")
                break

    def stop(self):
        """Stop the streaming process."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.stream_thread:
            self.stream_thread.join(timeout=1)
