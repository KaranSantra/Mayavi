# asr/asr_module.py
import whisperx
import pyaudio
import numpy as np
import threading
import queue
import time
import gc
import socket
import json
import re
import argparse


class ASRModule:
    def __init__(
        self,
        host="127.0.0.1",
        port=5000,
        device="cpu",
        model_name="tiny.en",
        compute_type="int8",
    ):
        self.device = device
        self.audio_queue = queue.Queue()
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.host = host
        self.port = port

        # Initialize whisperX model
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)

        # Audio recording parameters
        self.CHUNK = 1024 * 3
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000

        # Parameters for sentence boundary detection
        self.buffer = ""
        self.silence_threshold = 1.0  # seconds of silence to consider a pause
        self.last_speech_time = time.time()
        self.buffer_timeout = (
            5.0  # seconds before sending buffer even without clear ending
        )

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

    def start_server(self):
        """Start a socket server to communicate with the main program"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"ASR Module: Socket server started at {self.host}:{self.port}")
            self.client_socket, addr = self.server_socket.accept()
            print(f"ASR Module: Connected to client at {addr}")
            return True
        except Exception as e:
            print(f"ASR Module: Socket error - {e}")
            return False

    def send_message(self, text):
        """Send transcribed text through the socket"""
        if self.client_socket:
            try:
                message = json.dumps({"type": "transcription", "text": text})
                self.client_socket.sendall(f"{message}\n".encode("utf-8"))
                print(f"ASR Module: Sent message: {text}")
            except Exception as e:
                print(f"ASR Module: Error sending message - {e}")

    def start_recording(self):
        """Start the ASR pipeline"""
        if not self.start_server():
            return False

        self.running = True

        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=None,  # Ensure we're using blocking mode
            )

            # Verify stream is open
            if not self.stream.is_active():
                print("ASR Module: Failed to open audio stream")
                return False

            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True  # Make thread daemon so it exits with main
            self.record_thread.start()

            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_audio)
            self.process_thread.daemon = (
                True  # Make thread daemon so it exits with main
            )
            self.process_thread.start()

            return True

        except Exception as e:
            print(f"ASR Module: Error initializing audio - {e}")
            self.running = False
            return False

    def _record_audio(self):
        """Record audio from microphone and add to queue"""
        while self.running:
            try:
                if not self.stream.is_active():
                    print("ASR Module: Stream inactive, stopping recording")
                    break

                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_chunk)

            except OSError as e:
                if not self.running:  # Expected error when stopping
                    break
                print(f"ASR Module: Stream error - {e}")
                break
            except Exception as e:
                print(f"ASR Module: Error recording audio - {e}")
                if not self.running:
                    break
                time.sleep(0.1)  # Prevent tight loop on error

    def _is_sentence_end(self, text):
        """Check if text likely ends a sentence"""
        # Check for ending punctuation
        if re.search(r"[.!?]\s*$", text):
            return True
        return False

    def _process_audio(self):
        """Process audio chunks and transcribe with WhisperX"""
        audio_buffer = []
        silence_start = time.time()

        while self.running:
            # Process audio in 3-second chunks
            if len(audio_buffer) * self.CHUNK / self.RATE >= 3.0:
                try:
                    # Convert buffer to numpy array
                    audio_data = np.concatenate(audio_buffer)

                    # Transcribe with whisperX
                    result = self.model.transcribe(audio_data, batch_size=1)

                    # Check if we got any transcription
                    if result["segments"]:
                        current_text = result["segments"][0]["text"].strip()

                        # If we have text, update last speech time
                        if current_text:
                            self.last_speech_time = time.time()
                            silence_start = time.time()

                            # Add to buffer
                            if not self.buffer:
                                self.buffer = current_text
                            else:
                                self.buffer += " " + current_text

                            # Check if the buffer has a sentence ending
                            if self._is_sentence_end(self.buffer):
                                self.send_message(self.buffer)
                                self.buffer = ""

                    # Check if we've been silent long enough to consider it a pause
                    current_time = time.time()
                    if self.buffer and (
                        current_time - silence_start >= self.silence_threshold
                    ):
                        self.send_message(self.buffer)
                        self.buffer = ""

                    # Check if buffer timeout has been reached
                    if self.buffer and (
                        current_time - self.last_speech_time >= self.buffer_timeout
                    ):
                        self.send_message(self.buffer)
                        self.buffer = ""

                    # Clear audio buffer
                    audio_buffer = []

                except Exception as e:
                    print(f"ASR Module: Error processing audio - {e}")
                    audio_buffer = []

            # Add new audio to buffer
            if not self.audio_queue.empty():
                audio_buffer.append(self.audio_queue.get())

            # Small sleep to reduce CPU usage
            time.sleep(0.01)

    def stop(self):
        """Stop the ASR pipeline"""
        if self.running:
            self.running = False
            time.sleep(0.2)  # Give threads time to see running=False

            # Stop audio stream
            if hasattr(self, "stream"):
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"ASR Module: Error closing stream - {e}")

            # Clean up PyAudio
            if hasattr(self, "p"):
                try:
                    self.p.terminate()
                except Exception as e:
                    print(f"ASR Module: Error terminating PyAudio - {e}")

            # Close sockets
            if self.client_socket:
                try:
                    self.client_socket.close()
                except Exception as e:
                    print(f"ASR Module: Error closing client socket - {e}")

            if self.server_socket:
                try:
                    self.server_socket.close()
                except Exception as e:
                    print(f"ASR Module: Error closing server socket - {e}")

            # Wait for threads to finish
            if hasattr(self, "record_thread") and self.record_thread:
                self.record_thread.join(timeout=2.0)

            if hasattr(self, "process_thread") and self.process_thread:
                self.process_thread.join(timeout=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Module")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()

    asr_module = ASRModule(port=args.port)
    print(f"Starting ASR Module on port {args.port}... Press Ctrl+C to stop")
    try:
        if asr_module.start_recording():
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping ASR Module...")
        asr_module.stop()
