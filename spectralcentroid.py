# The spectral centroid is a measure of a sound's "brightness" or "richness," representing the "center of mass" or average frequency of its spectrum,
# calculated as a weighted average of its frequency components.

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

# spectral_centroid is 2D: (1, num_frames)
# Get mean value
mean_spectral_centroid = np.mean(spectral_centroid)

print(f"Spectral Centroid: {mean_spectral_centroid:.2f} Hz")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")

# Interpretation
if mean_spectral_centroid > 3000:
    print("→ Bright sound (high frequencies dominant)")
elif mean_spectral_centroid > 2000:
    print("→ Moderately bright")
else:
    print("→ Dark/warm sound (low frequencies dominant)")
