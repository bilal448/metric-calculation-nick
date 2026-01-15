# Spectral rolloff is a measure used in digital signal processing that defines the frequency below which a specified proportion
# of the total spectral energy is contained. It provides information about the shape and skewness of the energy distribution
# in a sound's frequency spectrum

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral rolloff (default is 85% of energy)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)

# Get mean value
mean_rolloff = np.mean(spectral_rolloff)

print(f"Spectral Rolloff: {mean_rolloff:.2f} Hz")