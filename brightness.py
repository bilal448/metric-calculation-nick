# The brightness of a sound refers to the prominence of its higher frequencies, making it sound crisp, clear,
# and detailed, like a cymbal or high-pitched voice

import librosa
import numpy as np

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral centroid (brightness)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

# Mean brightness
mean_centroid = np.mean(spectral_centroid)

# Normalize to 0-1 (brightness score)
brightness = mean_centroid / (sr / 2)

print(f"Spectral Centroid: {mean_centroid:.2f} Hz")
print(f"Brightness (0-1): {brightness:.4f}")