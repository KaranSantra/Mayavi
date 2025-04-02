import numpy as np
import whisperx
import logging
import gc
from functools import lru_cache
import torch
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_whisper_model(model_name="base.en", device="cpu", compute_type="int8"):
    """Get or create a singleton instance of the WhisperX model."""
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        logger.info(f"ASR model loaded successfully: {model_name} on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise


class ASR:
    def __init__(self, model_name="base.en", device="cpu", compute_type="int8"):
        """Initialize the ASR model."""
        self.model = get_whisper_model(model_name, device, compute_type)
        self.device = device
        self.target_sample_rate = 16000

    def preprocess_audio(self, audio_data):
        """Preprocess audio data for transcription."""
        # Ensure audio data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1] range if needed
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        # Ensure audio is writable
        audio_data = np.copy(audio_data)

        # Apply pre-emphasis filter to enhance speech frequencies
        pre_emphasis = 0.97
        emphasized_audio = np.append(
            audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1]
        )

        return emphasized_audio

    def transcribe(self, audio_data):
        """
        Transcribe audio data to text using WhisperX.

        Args:
            audio_data (np.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text
        """
        try:
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data)

            # Transcribe with whisperX
            result = self.model.transcribe(audio_data, batch_size=1)

            # Check if any speech was detected
            if not result["segments"]:
                logger.warning("No speech segments detected in audio")
                return ""

            # Align whisper output
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=self.device
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    self.device,
                    return_char_alignments=False,
                )

                # Clean up alignment model
                del model_a
                gc.collect()
            except Exception as e:
                logger.warning(f"Alignment failed, using unaligned transcription: {e}")
                # If alignment fails, use unaligned transcription
                result = {"segments": result["segments"]}

            # Extract text from segments
            transcript = " ".join(segment["text"] for segment in result["segments"])
            return transcript.strip()

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
