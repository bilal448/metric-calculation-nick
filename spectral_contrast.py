# Spectral contrast of sound refers to the difference in energy between the loudest (peaks)
# and softest (valleys) parts of a sound's frequency spectrum

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Single mean value across all bands and frames
mean_spectral_contrast = float(np.mean(spectral_contrast))

print(f"Mean Spectral Contrast: {mean_spectral_contrast:.2f} dB")