# Roughness of sound (or auditory roughness) is a psychoacoustic perception of a sound's harshness or irregularity,
# caused by rapid fluctuations in sound level (amplitude modulation, typically 20-200 Hz) or closely spaced, dissonant
# frequency components (partials) that the ear struggles to separate, creating a "buzzy" or "grainy" texture, crucial in music,
# voice quality, and noise assessment.



import librosa
import numpy as np
from scipy.signal import find_peaks

audio_path = "trimmed_output33.wav"


def hz_to_bark(freq):
    """Convert frequency to Bark scale (critical bands)"""
    return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)


y, sr = librosa.load(audio_path, sr=None)

# Get spectrum
stft = librosa.stft(y)
magnitude = np.mean(np.abs(stft), axis=1)
freqs = librosa.fft_frequencies(sr=sr)

# Find spectral peaks (partials)
peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.05)
peak_freqs = freqs[peaks]
peak_mags = magnitude[peaks]

# Convert to Bark scale
peak_barks = hz_to_bark(peak_freqs)

# Calculate roughness between all pairs
roughness = 0
pair_count = 0

for i in range(len(peak_freqs)):
    for j in range(i + 1, len(peak_freqs)):
        # Distance in critical bands
        bark_distance = abs(peak_barks[i] - peak_barks[j])

        # Roughness occurs when partials are < 1 critical band apart
        if bark_distance < 1.2:  # Within critical band
            # Weight by amplitude (similar amplitudes = more roughness)
            avg_mag = (peak_mags[i] + peak_mags[j]) / 2
            mag_product = peak_mags[i] * peak_mags[j]

            # Roughness function (peaks around 0.5 Bark)
            rough_contribution = mag_product * np.exp(-bark_distance)
            roughness += rough_contribution
            pair_count += 1

# Normalize
roughness = roughness / (len(peak_freqs) ** 2) if len(peak_freqs) > 1 else 0

print(f"Roughness: {roughness:.6f}")
print(f"Close pairs: {pair_count}")