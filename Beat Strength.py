# In acoustics, the "beat strength" of a sound refers to the magnitude of the periodic fluctuations in loudness that occur when two sound waves of slightly
# different frequencies interfere with each other. It is not a standard scientific term but describes the perceived intensity of the beating effect.

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Detect beats
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Get onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Extract beat strength at each detected beat
beat_strengths = onset_env[beat_frames]

# Beat strength metrics
avg_beat_strength = np.mean(beat_strengths)
max_beat_strength = np.max(beat_strengths)
min_beat_strength = np.min(beat_strengths)
beat_strength_consistency = np.std(beat_strengths)

print(f"Average Beat Strength: {avg_beat_strength:.4f}")
print(f"Max Beat Strength: {max_beat_strength:.4f}")
print(f"Min Beat Strength: {min_beat_strength:.4f}")
print(f"Beat Consistency (lower = more consistent): {beat_strength_consistency:.4f}")