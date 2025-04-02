import socket
import threading
import numpy as np
import logging
import json
from asr import ASR
from llm import LLMModule
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Server:
    def __init__(self, host="0.0.0.0", port=5000, llm_port=5001):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None

        # Initialize ASR with appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.asr = ASR(model_name="base.en", device=device, compute_type=compute_type)

        # Initialize LLM with appropriate device
        llm_device = "cuda" if torch.cuda.is_available() else "cpu"
        llm_compute_type = "float16" if llm_device == "cuda" else "int8"
        self.llm = LLMModule(
            host="localhost",
            port=llm_port,
            device=llm_device,
            compute_type=llm_compute_type,
        )

        self.audio_buffer = []
        self.buffer_size = 16000 * 3  # 3 seconds of audio at 16kHz
        self.client_threads = []

    def start(self):
        """Start the server and listen for connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            logger.info(f"Server started on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"New connection from {address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except socket.error as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                    break

        except socket.error as e:
            logger.error(f"Failed to start server: {e}")
            raise
        finally:
            self.stop()

    def handle_client(self, client_socket):
        """Handle a client connection."""
        audio_buffer = []
        try:
            while self.running:
                try:
                    # Receive audio chunk
                    data = client_socket.recv(4096)  # Adjust buffer size as needed
                    if not data:
                        break

                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    audio_buffer.extend(audio_chunk)

                    # Process audio when buffer is full
                    if len(audio_buffer) >= self.buffer_size:
                        audio_data = np.array(audio_buffer[: self.buffer_size])
                        transcript = self.process_audio(audio_data)

                        # Send transcript to LLM and get response
                        if transcript:
                            # Send transcript to client
                            transcript_data = {
                                "type": "transcription",
                                "text": transcript,
                            }
                            client_socket.send(
                                f"{json.dumps(transcript_data)}\n".encode()
                            )
                            logger.info(f"Sent transcript: {transcript}")

                            # Get LLM response
                            llm_response = self.llm.generate_response(transcript)
                            if llm_response:
                                response_data = {
                                    "type": "llm_response",
                                    "text": llm_response,
                                }
                                client_socket.send(
                                    f"{json.dumps(response_data)}\n".encode()
                                )
                                logger.info(f"Sent LLM response: {llm_response}")

                        # Keep remaining audio in buffer
                        audio_buffer = audio_buffer[self.buffer_size :]

                except socket.error as e:
                    logger.error(f"Error receiving data: {e}")
                    break

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def process_audio(self, audio_data):
        """Process audio data and return transcript."""
        try:
            return self.asr.transcribe(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return ""

    def stop(self):
        """Stop the server and clean up resources."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

        # Stop LLM module
        self.llm.stop()

        # Wait for client threads to finish
        for thread in self.client_threads:
            thread.join(timeout=1)
        self.client_threads.clear()


if __name__ == "__main__":
    # Create and start server
    server = Server()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.stop()
