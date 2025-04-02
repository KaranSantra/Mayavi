import numpy as np


class PauseDetector:
    def __init__(self, energy_threshold=0.1, pause_frames=3):
        self.energy_threshold = energy_threshold
        self.pause_frames = pause_frames
        self.low_energy_frames = 0
        self.is_speaking = True

    def process_chunk(self, audio_chunk):
        """Process an audio chunk and detect if there's a pause in speech."""
        # Calculate energy of the chunk
        energy = np.mean(np.abs(audio_chunk))

        if energy < self.energy_threshold:
            self.low_energy_frames += 1
            if self.low_energy_frames >= self.pause_frames:
                self.is_speaking = False
        else:
            self.low_energy_frames = 0
            self.is_speaking = True

    def is_paused(self):
        """Return True if speech is currently paused."""
        return not self.is_speaking

    def reset(self):
        """Reset the pause detector state."""
        self.low_energy_frames = 0
        self.is_speaking = True
