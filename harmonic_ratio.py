# A harmonic ratio in sound describes the mathematical relationship between frequencies in the harmonic series, where higher tones are simple
# integer multiples (like 2:1 for an octave, 3:2 for a perfect fifth) of the fundamental frequency, creating consonance; in audio analysis,
# it also refers to the Harmonic-to-Noise Ratio (HNR), quantifying the periodic (harmonic) energy versus random noise in a sound, expressed in dB.

import librosa
import numpy as np


audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Separate harmonic and percussive (noise) components
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Calculate power (energy)
harmonic_power = np.sum(y_harmonic ** 2)
noise_power = np.sum(y_percussive ** 2)

# HNR in dB
hnr = 10 * np.log10(harmonic_power / (noise_power + 1e-10))

print(f"Harmonic-to-Noise Ratio: {hnr:.2f} dB")