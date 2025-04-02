import unittest
import socket
import time
import subprocess
import sys
import os
import json
from pathlib import Path
import numpy as np


class TestRemoteIntegration(unittest.TestCase):
    def setUp(self):
        # These should be configured based on your cloud VM
        self.server_host = "35.192.131.234"  # Replace with your cloud VM IP
        self.server_port = 5000
        self.test_duration = 10  # seconds

        # Path to the server and client scripts
        self.base_path = Path(__file__).parent.parent
        self.server_script = self.base_path / "server.py"
        self.client_script = self.base_path / "client.py"

    def test_server_connection(self):
        """Test if we can establish a connection to the server"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.server_host, self.server_port))
            self.assertTrue(True, "Successfully connected to server")
        except Exception as e:
            self.fail(f"Failed to connect to server: {e}")
        finally:
            sock.close()

    def test_audio_transmission(self):
        """Test audio transmission from client to server"""
        # Create a test audio file (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)

        # Try to connect and send audio
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.server_host, self.server_port))

            # Send audio data with header
            header = {
                "type": "audio",
                "length": len(test_audio.tobytes()),
                "sample_rate": 16000,
            }
            header_bytes = json.dumps(header).encode("utf-8")
            header_length = len(header_bytes).to_bytes(4, byteorder="big")

            # Send header length, header, then audio data
            sock.send(header_length)
            sock.send(header_bytes)
            sock.send(test_audio.tobytes())

            # Wait for response
            response_length = int.from_bytes(sock.recv(4), byteorder="big")
            response_bytes = sock.recv(response_length)
            response = json.loads(response_bytes.decode("utf-8"))

            self.assertIn("status", response, "Response should contain status field")
            self.assertEqual(
                response["status"], "success", "Server should return success status"
            )

        except Exception as e:
            self.fail(f"Failed to transmit audio: {e}")
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
