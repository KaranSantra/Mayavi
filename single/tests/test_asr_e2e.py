import unittest
import os
import wave
import numpy as np
from client import Client
from asr import ASR
from audio_streamer import AudioStreamer
import time
import logging
from scipy import signal
import re
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestASRE2E(unittest.TestCase):
    def setUp(self):
        self.samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
        self.asr = ASR()
        self.streamer = AudioStreamer()
        self.client = Client()
        self.target_sample_rate = 16000
        self.similarity_threshold = 0.85  # 85% similarity threshold

    def normalize_text(self, text):
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def get_text_similarity(self, text1, text2):
        """Get similarity ratio between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()

    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio_data

        # Calculate number of samples for target length
        target_length = int(len(audio_data) * float(target_sr) / orig_sr)
        resampled = signal.resample(audio_data, target_length)
        return resampled

    def load_audio_file(self, wav_file):
        """Load and preprocess audio file."""
        with wave.open(wav_file, "rb") as wf:
            # Get audio parameters
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()

            logger.info(
                f"Audio file parameters: {n_frames} frames, {n_channels} channels, "
                f"{sampwidth} bytes per sample, {framerate} Hz"
            )

            # Read audio data
            audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Reshape if stereo
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                # Convert to mono by averaging channels
                audio_data = np.mean(audio_data, axis=1)

            # Resample to target sample rate if needed
            if framerate != self.target_sample_rate:
                logger.info(
                    f"Resampling from {framerate}Hz to {self.target_sample_rate}Hz"
                )
                audio_data = self.resample_audio(
                    audio_data, framerate, self.target_sample_rate
                )

            return audio_data

    def test_transcribe_sample_1(self):
        """Test transcription of sample_audio_1.wav"""
        wav_file = os.path.join(self.samples_dir, "sample_audio_1.wav")
        txt_file = os.path.join(self.samples_dir, "sample_audio_1.txt")

        # Read expected transcript
        with open(txt_file, "r") as f:
            expected_transcript = f.read().strip()

        # Load and process audio
        audio_data = self.load_audio_file(wav_file)

        # Process in chunks to simulate real-time streaming
        chunk_size = self.client.CHUNK
        chunks = [
            audio_data[i : i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        # Process each chunk
        for chunk in chunks:
            if len(chunk) == chunk_size:  # Only send full chunks
                self.streamer.send_chunk(chunk)

        # Get final transcript
        transcript = self.asr.transcribe(audio_data)

        # Compare transcripts
        self.assertIsInstance(transcript, str)
        self.assertTrue(len(transcript) > 0)

        # Normalize texts
        norm_expected = self.normalize_text(expected_transcript)
        norm_transcript = self.normalize_text(transcript)

        # Log both versions
        logger.info(f"Expected (normalized): {norm_expected}")
        logger.info(f"Got (normalized): {norm_transcript}")

        # Get similarity ratio
        similarity = self.get_text_similarity(norm_expected, norm_transcript)
        logger.info(f"Similarity ratio: {similarity:.2f}")

        # Assert similarity is above threshold
        self.assertGreaterEqual(
            similarity,
            self.similarity_threshold,
            f"Transcript similarity {similarity:.2f} is below threshold {self.similarity_threshold}",
        )

    def test_transcribe_sample_2(self):
        """Test transcription of sample_audio_2.wav"""
        wav_file = os.path.join(self.samples_dir, "sample_audio_2.wav")
        txt_file = os.path.join(self.samples_dir, "sample_audio_2.txt")

        # Read expected transcript
        with open(txt_file, "r") as f:
            expected_transcript = f.read().strip()

        # Load and process audio
        audio_data = self.load_audio_file(wav_file)

        # Process in chunks to simulate real-time streaming
        chunk_size = self.client.CHUNK
        chunks = [
            audio_data[i : i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        # Process each chunk
        for chunk in chunks:
            if len(chunk) == chunk_size:  # Only send full chunks
                self.streamer.send_chunk(chunk)

        # Get final transcript
        transcript = self.asr.transcribe(audio_data)

        # Compare transcripts
        self.assertIsInstance(transcript, str)
        self.assertTrue(len(transcript) > 0)

        # Normalize texts
        norm_expected = self.normalize_text(expected_transcript)
        norm_transcript = self.normalize_text(transcript)

        # Log both versions
        logger.info(f"Expected (normalized): {norm_expected}")
        logger.info(f"Got (normalized): {norm_transcript}")

        # Get similarity ratio
        similarity = self.get_text_similarity(norm_expected, norm_transcript)
        logger.info(f"Similarity ratio: {similarity:.2f}")

        # Assert similarity is above threshold
        self.assertGreaterEqual(
            similarity,
            self.similarity_threshold,
            f"Transcript similarity {similarity:.2f} is below threshold {self.similarity_threshold}",
        )

    def test_transcribe_sample_3(self):
        """Test transcription of sample_audio_3.wav"""
        wav_file = os.path.join(self.samples_dir, "sample_audio_3.wav")
        txt_file = os.path.join(self.samples_dir, "sample_audio_3.txt")

        # Read expected transcript
        with open(txt_file, "r") as f:
            expected_transcript = f.read().strip()

        # Load and process audio
        audio_data = self.load_audio_file(wav_file)

        # Process in chunks to simulate real-time streaming
        chunk_size = self.client.CHUNK
        chunks = [
            audio_data[i : i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        # Process each chunk
        for chunk in chunks:
            if len(chunk) == chunk_size:  # Only send full chunks
                self.streamer.send_chunk(chunk)

        # Get final transcript
        transcript = self.asr.transcribe(audio_data)

        # Compare transcripts
        self.assertIsInstance(transcript, str)
        self.assertTrue(len(transcript) > 0)

        # Normalize texts
        norm_expected = self.normalize_text(expected_transcript)
        norm_transcript = self.normalize_text(transcript)

        # Log both versions
        logger.info(f"Expected (normalized): {norm_expected}")
        logger.info(f"Got (normalized): {norm_transcript}")

        # Get similarity ratio
        similarity = self.get_text_similarity(norm_expected, norm_transcript)
        logger.info(f"Similarity ratio: {similarity:.2f}")

        # Assert similarity is above threshold
        self.assertGreaterEqual(
            similarity,
            self.similarity_threshold,
            f"Transcript similarity {similarity:.2f} is below threshold {self.similarity_threshold}",
        )


if __name__ == "__main__":
    unittest.main()
