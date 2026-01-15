# Zero-Crossing Rate (ZCR) measures how often a sound wave's amplitude changes from positive to
# negative (or vice-versa) within a specific time frame,


import librosa
import numpy as np

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate zero-crossing rate
zcr = librosa.feature.zero_crossing_rate(y)

# Get mean value
mean_zcr = np.mean(zcr)

print(zcr)
print(f"Mean Zero-Crossing Rate: {mean_zcr:.4f}")