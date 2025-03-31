# asr/asr_module.py
import whisperx
import numpy as np
import threading
import queue
import time
import gc
import socket
import json
import re
import argparse
import torch


class ASRModule:
    def __init__(
        self,
        host="127.0.0.1",
        port=5000,
        model_name="tiny.en",
    ):
        # Handle device and compute type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.audio_queue = queue.Queue()
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.host = host
        self.port = port

        # Initialize whisperX model
        self.model = whisperx.load_model(
            model_name, self.device, compute_type=self.compute_type
        )

        # Parameters for sentence boundary detection
        self.buffer = ""
        self.silence_threshold = 1.0  # seconds of silence to consider a pause
        self.last_speech_time = time.time()
        self.buffer_timeout = (
            5.0  # seconds before sending buffer even without clear ending
        )

        # Audio parameters
        self.RATE = 16000

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

    def start_processing(self):
        """Start the ASR pipeline without microphone recording"""
        if not self.start_server():
            return False

        self.running = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio_from_socket)
        self.process_thread.daemon = True
        self.process_thread.start()

        return True

    def _process_audio_from_socket(self):
        """Process audio chunks received from socket and transcribe with WhisperX"""
        audio_buffer = []
        buffer_duration = 0
        target_duration = 3.0  # Process 3 seconds of audio at a time
        silence_start = time.time()

        while self.running:
            try:
                # Wait for data from the server
                data = self.client_socket.recv(4096)
                if not data:
                    time.sleep(0.1)
                    continue

                # Convert binary data to numpy array (assuming float32 format)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Add to buffer
                audio_buffer.append(audio_chunk)

                # Update buffer duration
                buffer_duration += len(audio_chunk) / self.RATE

                # Process buffer when we have enough data
                if buffer_duration >= target_duration:
                    # Concatenate audio chunks
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

                    # Reset buffer
                    audio_buffer = []
                    buffer_duration = 0

            except Exception as e:
                print(f"ASR Module: Error processing audio - {e}")
                audio_buffer = []
                buffer_duration = 0
                time.sleep(0.1)  # Prevent tight loop on error

    def _is_sentence_end(self, text):
        """Check if text likely ends a sentence"""
        # Check for ending punctuation
        if re.search(r"[.!?]\s*$", text):
            return True
        return False

    def stop(self):
        """Stop the ASR pipeline"""
        if self.running:
            self.running = False
            time.sleep(0.2)  # Give threads time to see running=False

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
            if hasattr(self, "process_thread") and self.process_thread:
                self.process_thread.join(timeout=5.0)

            # Clean up CUDA memory if using GPU
            if hasattr(self, "model") and self.device == "cuda":
                try:
                    del self.model
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"ASR Module: Error cleaning up CUDA memory - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Module")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument(
        "--model_name",
        type=str,
        default="tiny.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v2"],
        help="WhisperX model to use (default: tiny.en)",
    )

    args = parser.parse_args()

    asr_module = ASRModule(
        port=args.port,
        model_name=args.model_name,
    )
    print(
        f"Starting ASR Module on port {args.port} with model {args.model_name}... Press Ctrl+C to stop"
    )
    try:
        if (
            asr_module.start_processing()
        ):  # Changed from start_recording to start_processing
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping ASR Module...")
        asr_module.stop()
