# Dissonance of sound refers to a harsh, clashing, or unstable combination of
# notes or sounds that creates tension

import librosa
import numpy as np
from scipy.signal import find_peaks

audio_path = "trimmed_output33.wav"
def hz_to_bark(freq):
    """Convert to critical bandwidth scale"""
    return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)


y, sr = librosa.load(audio_path, sr=None)

# Get spectrum
stft = librosa.stft(y)
magnitude = np.mean(np.abs(stft), axis=1)
freqs = librosa.fft_frequencies(sr=sr)

# Find peaks (partials)
peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.05)
peak_freqs = freqs[peaks]
peak_mags = magnitude[peaks]

# Convert to Bark
peak_barks = hz_to_bark(peak_freqs)

# Calculate dissonance
dissonance = 0

for i in range(len(peak_freqs)):
    for j in range(i + 1, len(peak_freqs)):
        bark_dist = abs(peak_barks[i] - peak_barks[j])

        # Dissonance curve (Plomp & Levelt)
        # Maximum dissonance around 0.25 critical bands
        x = bark_dist * 1.2
        diss = np.exp(-3.5 * x) - np.exp(-5.75 * x)

        # Weight by amplitude
        weight = min(peak_mags[i], peak_mags[j])
        dissonance += weight * diss

# Normalize
dissonance = dissonance / len(peak_freqs) if len(peak_freqs) > 0 else 0

print(f"Dissonance: {dissonance:.6f}")