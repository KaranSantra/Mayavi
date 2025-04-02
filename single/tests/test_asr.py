import unittest
import numpy as np
from asr import ASR
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestASR(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.asr = ASR(model_name="tiny.en")  # Use tiny model for faster tests

    def test_empty_audio(self):
        """Test handling of empty audio input."""
        empty_audio = np.array([], dtype=np.float32)
        transcript = self.asr.transcribe(empty_audio)
        self.assertEqual(transcript, "")

    def test_silent_audio(self):
        """Test handling of silent audio input."""
        silent_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        transcript = self.asr.transcribe(silent_audio)
        self.assertEqual(transcript, "")

    def test_invalid_audio_type(self):
        """Test handling of invalid audio data type."""
        invalid_audio = np.array([1, 2, 3], dtype=np.int32)
        transcript = self.asr.transcribe(invalid_audio)
        self.assertEqual(transcript, "")

    def test_audio_normalization(self):
        """Test audio normalization in preprocessing."""
        # Create audio with values > 1
        audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        processed = self.asr.preprocess_audio(audio)
        self.assertTrue(np.all(np.abs(processed) <= 1.0))

    def test_short_audio(self):
        """Test handling of very short audio input."""
        short_audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        transcript = self.asr.transcribe(short_audio)
        self.assertEqual(transcript, "")

    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        with self.assertRaises(Exception):
            ASR(model_name="invalid_model")

    def test_audio_preprocessing(self):
        """Test audio preprocessing steps."""
        # Create test audio
        audio = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)

        # Test preprocessing
        processed = self.asr.preprocess_audio(audio)

        # Check if output is float32
        self.assertEqual(processed.dtype, np.float32)

        # Check if pre-emphasis was applied (first value should be unchanged)
        self.assertEqual(processed[0], audio[0])

    def test_transcribe_with_noise(self):
        """Test transcription with noisy audio."""
        # Create noisy audio
        noise = np.random.normal(0, 0.1, 16000).astype(np.float32)
        transcript = self.asr.transcribe(noise)
        self.assertIsInstance(transcript, str)

    def test_model_caching(self):
        """Test that model is properly cached."""
        # Create two ASR instances with same parameters
        asr1 = ASR(model_name="tiny.en")
        asr2 = ASR(model_name="tiny.en")

        # Check if they share the same model instance
        self.assertIs(asr1.model, asr2.model)


if __name__ == "__main__":
    unittest.main()
