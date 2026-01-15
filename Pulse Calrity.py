# Pulse clarity is considered as a high-level musical dimen- sion that conveys how easily in a given musical piece, or a particular moment during that piece,
# listeners can perceive the underlying rhythmic or metrical pulsation. The objective of this study is to establish a composite model explaining pulse clarity
# judgments from the analysis of audio record- ings

import librosa
import numpy as np


def calculate_pulse_clarity_simple(audio_path):
    """
    Simplified version focusing on key research insights
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env = onset_env / np.max(onset_env) if np.max(onset_env) > 0 else onset_env

    # Autocorrelation (periodicity)
    autocorr = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr

    # Tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempogram_mean = np.mean(tempogram, axis=1)
    tempogram_norm = tempogram_mean / np.sum(tempogram_mean) if np.sum(tempogram_mean) > 0 else tempogram_mean

    # Combine features
    periodicity = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
    tempo_peak = np.max(tempogram_norm)

    # Weighted average (based on Lartillot et al.)
    pulse_clarity = 0.6 * periodicity + 0.4 * tempo_peak

    return float(pulse_clarity)


# Usage
pulse_clarity = calculate_pulse_clarity_simple("trimmed_output33.wav")
print(f"Pulse Clarity: {pulse_clarity:.4f}")