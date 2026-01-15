#The crest factor of sound is the ratio of a sound wave's peak (maximum) amplitude to its RMS (Root Mean Square, or average) value, indicating how "spiky" or dynamic the signal is, expressed in decibels (dB)

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Calculate peak amplitude
peak = np.max(np.abs(y))

# Calculate RMS
rms = np.sqrt(np.mean(y**2))

# Crest factor (ratio)
crest_factor = peak / rms

# Crest factor in dB
crest_factor_db = 20 * np.log10(crest_factor)

print(f"Peak: {peak:.4f}")
print(f"RMS: {rms:.4f}")
print(f"Crest Factor: {crest_factor:.2f}")
print(f"Crest Factor (dB): {crest_factor_db:.2f} dB")