import unittest
from unittest.mock import Mock, patch
import json
import socket
import threading
import time
from llm import LLMModule


class TestLLMModule(unittest.TestCase):
    @patch("llm.AutoModelForCausalLM")
    @patch("llm.AutoTokenizer")
    def setUp(self, mock_tokenizer, mock_model):
        """Set up test fixtures."""
        # Mock tokenizer
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 0
        mock_tokenizer.return_value.decode.return_value = "Mocked response"
        mock_tokenizer.return_value.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }

        # Mock model
        mock_model.return_value.generate.return_value = Mock()
        mock_model.return_value.to.return_value = mock_model.return_value

        self.llm = LLMModule(host="127.0.0.1", port=5001)
        self.mock_socket = Mock()
        self.llm.client_socket = self.mock_socket
        self.llm.server_socket = Mock()

    def test_format_prompt(self):
        """Test prompt formatting with chat history."""
        # Test with empty history
        prompt = self.llm.format_prompt("Hello")
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)

        # Test with history
        self.llm.history = [
            {"user": "Hi", "assistant": "Hello!"},
            {"user": "How are you?", "assistant": "I'm good, thanks!"},
        ]
        prompt = self.llm.format_prompt("What's the weather?")
        self.assertIn("Hi", prompt)
        self.assertIn("How are you?", prompt)
        self.assertIn("What's the weather?", prompt)

    def test_send_response(self):
        """Test sending response through socket."""
        response = "Hello, how can I help you?"
        self.llm.send_response(response)
        self.mock_socket.sendall.assert_called_once()
        sent_data = self.mock_socket.sendall.call_args[0][0].decode("utf-8")
        self.assertIn(response, sent_data)

    def test_process_messages(self):
        """Test processing incoming messages."""
        # Create a test message
        test_message = {"type": "transcription", "text": "Hello, how are you?"}
        message_data = f"{json.dumps(test_message)}\n".encode("utf-8")

        # Mock socket receive
        self.mock_socket.recv.side_effect = [message_data, b""]

        # Mock generate_response
        self.llm.generate_response = Mock(return_value="Test response")

        # Start processing in a separate thread
        self.llm.running = True
        process_thread = threading.Thread(target=self.llm.process_messages)
        process_thread.daemon = True
        process_thread.start()

        # Wait for processing
        time.sleep(0.1)

        # Stop the thread
        self.llm.running = False
        process_thread.join(timeout=1)

        # Verify response was sent
        self.mock_socket.sendall.assert_called_once()

    def test_start_server(self):
        """Test server startup."""
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.accept.return_value = (Mock(), ("127.0.0.1", 5001))
            success = self.llm.start_server()
            self.assertTrue(success)
            mock_socket.return_value.bind.assert_called_once()
            mock_socket.return_value.listen.assert_called_once()

    def test_stop(self):
        """Test stopping the LLM module."""
        self.llm.running = True
        self.llm.stop()
        self.assertFalse(self.llm.running)
        self.mock_socket.close.assert_called_once()

    def test_error_handling(self):
        """Test error handling in message processing."""
        # Mock socket to raise an exception
        self.mock_socket.recv.side_effect = Exception("Test error")

        # Start processing in a separate thread
        self.llm.running = True
        process_thread = threading.Thread(target=self.llm.process_messages)
        process_thread.daemon = True
        process_thread.start()

        # Wait for processing
        time.sleep(0.1)

        # Stop the thread
        self.llm.running = False
        process_thread.join(timeout=1)

        # Verify the module stopped gracefully
        self.assertFalse(self.llm.running)

    def test_chat_history_management(self):
        """Test chat history management."""
        # Mock tokenizer decode to return a specific response
        self.llm.tokenizer.decode.return_value = "<|assistant|>Test response"

        # Test adding to history
        self.llm.history = []
        user_input = "Hello"
        response = self.llm.generate_response(user_input)

        self.assertEqual(response, "Test response")
        self.assertEqual(len(self.llm.history), 1)
        self.assertEqual(self.llm.history[0]["user"], user_input)
        self.assertEqual(self.llm.history[0]["assistant"], "Test response")

        # Test history limit
        for i in range(10):
            self.llm.generate_response(f"Message {i}")
        self.assertEqual(len(self.llm.history), 4)  # Only keeps last 4 conversations


if __name__ == "__main__":
    unittest.main()
