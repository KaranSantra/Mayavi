import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pyaudio
import wave
import threading
import queue
import time
from client import Client
from pause_detector import PauseDetector
from audio_streamer import AudioStreamer
from asr import ASR


class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_client_initialization(self):
        self.assertIsNotNone(self.client.p)
        self.assertEqual(self.client.CHUNK, 1024 * 3)
        self.assertEqual(self.client.FORMAT, pyaudio.paFloat32)
        self.assertEqual(self.client.CHANNELS, 1)
        self.assertEqual(self.client.RATE, 16000)
        self.assertFalse(self.client.running)
        self.assertIsInstance(self.client.audio_queue, queue.Queue)

    @patch("pyaudio.PyAudio")
    def test_start_recording(self, mock_pyaudio):
        # Setup mock
        mock_stream = MagicMock()
        mock_stream.read.return_value = np.ones(1024 * 3, dtype=np.float32)
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_instance.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pyaudio_instance

        # Create client with mocked PyAudio
        client = Client()
        client.p = mock_pyaudio_instance

        # Start recording
        client.start_recording()

        # Give the thread a moment to start
        time.sleep(0.1)

        # Verify
        self.assertTrue(client.running)
        mock_pyaudio_instance.open.assert_called_once()
        self.assertIsInstance(client.record_thread, threading.Thread)
        self.assertTrue(client.record_thread.is_alive())

        # Clean up
        client.stop()

    def test_stop_recording(self):
        self.client.running = True
        self.client.stream = MagicMock()
        self.client.record_thread = threading.Thread(target=lambda: None)
        self.client.record_thread.start()

        self.client.stop()

        self.assertFalse(self.client.running)
        self.client.stream.stop_stream.assert_called_once()
        self.client.stream.close.assert_called_once()


class TestPauseDetector(unittest.TestCase):
    def setUp(self):
        self.pause_detector = PauseDetector()

    def test_pause_detection(self):
        # Simulate audio data with different energy levels
        high_energy = np.ones(1024 * 3) * 0.8
        low_energy = np.ones(1024 * 3) * 0.1

        # Test continuous speech
        for _ in range(5):
            self.pause_detector.process_chunk(high_energy)
        self.assertFalse(self.pause_detector.is_paused())

        # Test pause
        for _ in range(3):
            self.pause_detector.process_chunk(low_energy)
        self.assertTrue(self.pause_detector.is_paused())

        # Test speech resumption
        self.pause_detector.process_chunk(high_energy)
        self.assertFalse(self.pause_detector.is_paused())


class TestAudioStreamer(unittest.TestCase):
    def setUp(self):
        self.streamer = AudioStreamer()

    def test_audio_chunk_sending(self):
        test_chunk = np.ones(1024 * 3)
        self.streamer.send_chunk(test_chunk)
        self.assertEqual(len(self.streamer.sent_chunks), 1)
        np.testing.assert_array_equal(self.streamer.sent_chunks[0], test_chunk)


class TestASR(unittest.TestCase):
    def setUp(self):
        self.asr = ASR()

    def test_transcription(self):
        # Mock audio data
        mock_audio = np.ones(16000 * 3)  # 3 seconds of audio at 16kHz

        # Test transcription
        transcript = self.asr.transcribe(mock_audio)
        self.assertIsInstance(transcript, str)
        self.assertTrue(len(transcript) > 0)


if __name__ == "__main__":
    unittest.main()
