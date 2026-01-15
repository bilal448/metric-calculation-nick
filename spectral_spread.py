# Spectral spread (or spectral variance) measures how dispersed a sound's energy is across frequencies, indicating its bandwidth and timbre
# Spectral Spread: Measures the deviation of frequencies from the centroid; large spread means broad frequency content.

import librosa
import numpy as np

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral bandwidth (this IS spectral spread)
spectral_spread = librosa.feature.spectral_bandwidth(y=y, sr=sr)

# Get mean value
mean_spread = np.mean(spectral_spread)

print(f"Spectral Spread: {mean_spread:.2f} Hz")