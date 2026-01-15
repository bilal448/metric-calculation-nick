# Spectral flatness: quantifying how tonal or noisy sound isSpectral flatness measures how "noise-like" versus "tone-like"
# a sound is by assessing the uniformity of its power spectrum, calculated as the ratio of the geometric mean to the arithmetic mean
# of the spectrum.

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate spectral flatness
spectral_flatness = librosa.feature.spectral_flatness(y=y)

# spectral_flatness is 2D: (1, num_frames)
# Get mean value
mean_flatness = np.mean(spectral_flatness)

print(f"Spectral Flatness: {mean_flatness:.4f}")
print(f"Shape: {spectral_flatness.shape}")

# Interpretation
if mean_flatness > 0.5:
    print("→ Noise-like (high flatness)")
elif mean_flatness > 0.2:
    print("→ Mixed (some noise, some tones)")
else:
    print("→ Tone-like (harmonic, musical)")