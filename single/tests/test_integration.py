import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import socket
import threading
import time
import numpy as np
import torch
from server import Server
from client import Client
from audio_streamer import AudioStreamer


class TestIntegration(unittest.TestCase):
    @patch("pyaudio.PyAudio")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model, mock_pyaudio):
        """Set up test fixtures."""
        # Mock PyAudio
        self.mock_stream = MagicMock()
        mock_pyaudio.return_value.open.return_value = self.mock_stream

        # Mock LLM components
        self.mock_tokenizer = mock_tokenizer.return_value
        self.mock_model = mock_model.return_value

        # Setup tokenizer mock
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Setup model mock
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_model.config = MagicMock()

        # Mock tokenizer decode to return a response
        self.mock_tokenizer.decode.return_value = (
            "I'm here to help! What can I assist you with?"
        )

        # Initialize components with test ports
        self.server_port = 5000
        self.llm_port = 5001

        # Create server instance (which creates LLM internally)
        self.server = Server(
            host="127.0.0.1", port=self.server_port, llm_port=self.llm_port
        )

        # Create client and streamer instances
        self.client = Client()
        self.streamer = AudioStreamer(host="127.0.0.1", port=self.server_port)

        # Mock ASR transcription
        self.mock_transcription = "Hello, how can you help me today?"

        # Mock audio data (3 seconds of silence at 16kHz)
        self.mock_audio = np.zeros(48000, dtype=np.float32)  # 3 seconds at 16kHz
        self.mock_stream.read.return_value = self.mock_audio.tobytes()

    @patch("asr.ASR.transcribe")
    def test_complete_flow(self, mock_transcribe):
        """Test the complete flow from client to server to LLM and back."""
        # Set up mock transcription
        mock_transcribe.return_value = self.mock_transcription

        # Start server in a separate thread
        server_thread = threading.Thread(target=self.server.start)
        server_thread.daemon = True
        server_thread.start()

        # Give some time for server to start
        time.sleep(2)

        # Start client and streamer
        self.client.start_recording()
        self.streamer.start_streaming()

        # Give some time for client and streamer to connect
        time.sleep(2)

        # Get audio chunk from client queue and send through streamer
        audio_chunk = self.client.audio_queue.get(timeout=1)
        self.streamer.send_chunk(audio_chunk)

        # Give more time for processing and LLM response
        time.sleep(5)

        # Verify transcription was called
        mock_transcribe.assert_called()

        # Verify LLM received the transcription and generated response
        self.assertGreater(len(self.server.llm.history), 0)
        self.assertEqual(self.server.llm.history[-1]["user"], self.mock_transcription)
        self.assertIsNotNone(self.server.llm.history[-1]["assistant"])
        self.assertEqual(
            self.server.llm.history[-1]["assistant"],
            "I'm here to help! What can I assist you with?",
        )

    def tearDown(self):
        """Clean up after tests."""
        # Stop all components
        self.client.stop()
        self.streamer.stop()
        self.server.stop()

        # Give some time for cleanup
        time.sleep(1)


if __name__ == "__main__":
    unittest.main()
