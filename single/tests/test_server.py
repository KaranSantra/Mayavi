import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import socket
import threading
import time
import random
from server import Server
from asr import ASR


class TestServer(unittest.TestCase):
    @patch("asr.ASR")
    def setUp(self, mock_asr):
        # Mock ASR
        self.mock_asr_instance = mock_asr.return_value
        self.mock_asr_instance.transcribe.return_value = "Test transcript"

        # Use a random port between 5000 and 6000
        self.port = random.randint(5000, 6000)
        self.server = Server(host="localhost", port=self.port)
        self.server.asr = self.mock_asr_instance

    def tearDown(self):
        if self.server:
            self.server.stop()

    def test_server_initialization(self):
        self.assertEqual(self.server.host, "localhost")
        self.assertEqual(self.server.port, self.port)
        self.assertFalse(self.server.running)
        self.assertIsInstance(self.server.asr, MagicMock)

    def test_start_server(self):
        # Start server in a separate thread
        server_thread = threading.Thread(target=self.server.start)
        server_thread.daemon = True
        server_thread.start()

        # Give the server time to start
        time.sleep(0.1)

        # Try to connect to verify server is running
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.connect(("localhost", self.port))
            test_socket.close()
            server_running = True
        except:
            server_running = False

        self.assertTrue(server_running)

        # Clean up
        self.server.stop()
        server_thread.join(timeout=1)

    def test_handle_client_connection(self):
        # Create test data
        test_audio = np.ones(self.server.buffer_size, dtype=np.float32)
        test_data = test_audio.tobytes()

        # Mock client socket
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [
            test_data,
            b"",
        ]  # Return data once, then empty to simulate disconnect

        # Set server as running
        self.server.running = True

        # Handle client directly
        self.server.handle_client(mock_socket)

        # Verify audio was processed
        mock_socket.recv.assert_called()
        self.mock_asr_instance.transcribe.assert_called_once()
        mock_socket.send.assert_called_once_with(b"Test transcript")

    def test_process_audio(self):
        # Create test audio data
        test_audio = np.random.normal(0, 0.1, self.server.buffer_size).astype(
            np.float32
        )

        # Process audio
        transcript = self.server.process_audio(test_audio)

        # Verify transcript
        self.assertEqual(transcript, "Test transcript")
        self.mock_asr_instance.transcribe.assert_called_once_with(test_audio)

    def test_error_handling(self):
        # Test server with invalid port
        with self.assertRaises(OverflowError):
            Server(host="localhost", port=-1).start()

        # Test server with invalid host
        with self.assertRaises(socket.error):
            Server(host="invalid_host", port=self.port).start()


if __name__ == "__main__":
    unittest.main()
