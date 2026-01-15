# Spectral flux is a measure of how quickly the power spectrum of a signal is changing, calculated by comparing the power spectrum for one frame against the power spectrum from the previous frame.[1]
#
# More precisely, it is usually calculated as the L2-norm (also known as the Euclidean distance) between the two normalised spectra. Calculated this way, the spectral flux is not dependent upon overall power (since the spectra are normalised), nor on phase considerations (since only the magnitudes are compared).
#
# The spectral flux can be used to determine the timbre of an audio signal, or in onset detection,[2] among other things.

import librosa
import numpy as np


def calculate_spectral_flux(audio_path):
    """
    Spectral Flux: L2-norm between consecutive normalized power spectra
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Power spectrum
    stft = librosa.stft(y)
    power_spec = np.abs(stft) ** 2

    # Normalize each frame
    power_spec_norm = power_spec / (np.sum(power_spec, axis=0, keepdims=True) + 1e-10)

    # L2-norm between consecutive frames
    diff = power_spec_norm[:, 1:] - power_spec_norm[:, :-1]
    spectral_flux = np.sqrt(np.sum(diff ** 2, axis=0))

    return {
        'mean': float(np.mean(spectral_flux)),
        'max': float(np.max(spectral_flux)),
        'std': float(np.std(spectral_flux))
    }


# Usage
result = calculate_spectral_flux("trimmed_output33.wav")
print(f"Mean Spectral Flux: {result['mean']:.4f}")