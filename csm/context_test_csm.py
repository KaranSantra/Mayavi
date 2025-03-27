from generator import load_csm_1b, Segment
import torchaudio
import torch

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cpu"


generator = load_csm_1b(device=device)


speakers = [1]
transcripts = [
    """But here's the twist. We've also got self-driving cars zipping through the air, robots doing all sorts of jobs, from crafting delicate jewelry to performing complex surgeries.
     Some people are thriving, while others are struggling to adapt."""
]
audio_paths = [
    "maya-speaking-15.wav",
]


def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor


segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="""You ever get that feeling, like the universe is holding its breath? 
    Like everything—every choice, every moment—has been leading to right now? Because I do.
     And Im telling you, if we step through that door, theres no going back. No safety net, no second chances. """,
    speaker=1,
    context=segments,
    max_audio_length_ms=20_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
