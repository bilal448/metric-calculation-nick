# The term "transient density of sound" refers to the sound energy per unit volume of a short-duration
# , high-amplitude acoustic event, which typically occurs at the beginning of a sound.

import librosa
import numpy as np


def calculate_transient_density_hpss(audio_path):
    """
    Transient density using HPSS method
    Measures what proportion of energy comes from transients
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Separate harmonic (sustained) and percussive (transient)
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)

    # Calculate RMS energy for each component
    rms_percussive = librosa.feature.rms(y=y_percussive)[0]
    rms_harmonic = librosa.feature.rms(y=y_harmonic)[0]
    rms_total = librosa.feature.rms(y=y)[0]

    # Mean energies
    mean_percussive_energy = np.mean(rms_percussive)
    mean_harmonic_energy = np.mean(rms_harmonic)
    mean_total_energy = np.mean(rms_total)

    # Transient density ratio (0-1)
    transient_ratio = mean_percussive_energy / mean_total_energy if mean_total_energy > 0 else 0
    harmonic_ratio = mean_harmonic_energy / mean_total_energy if mean_total_energy > 0 else 0

    # Alternative: Energy-based ratio using total power
    percussive_power = np.sum(y_percussive ** 2)
    total_power = np.sum(y ** 2)
    transient_energy_ratio = percussive_power / total_power if total_power > 0 else 0

    return {
        'transient_energy_ratio': float(transient_ratio),  # RMS-based
        'harmonic_energy_ratio': float(harmonic_ratio),  # Complement
        'transient_power_ratio': float(transient_energy_ratio),  # Power-based
        'mean_percussive_rms': float(mean_percussive_energy),
        'mean_harmonic_rms': float(mean_harmonic_energy)
    }


# Usage
metrics = calculate_transient_density_hpss("trimmed_output33.wav")

print("=" * 60)
print("TRANSIENT DENSITY (HPSS Energy Ratio Method)")
print("=" * 60)
print(
    f"Transient Energy Ratio: {metrics['transient_energy_ratio']:.4f} ({metrics['transient_energy_ratio'] * 100:.1f}%)")
print(f"Harmonic Energy Ratio: {metrics['harmonic_energy_ratio']:.4f} ({metrics['harmonic_energy_ratio'] * 100:.1f}%)")
print(f"Power-based Ratio: {metrics['transient_power_ratio']:.4f}")
print()
print(f"Mean Percussive RMS: {metrics['mean_percussive_rms']:.4f}")
print(f"Mean Harmonic RMS: {metrics['mean_harmonic_rms']:.4f}")
print("=" * 60)

# Interpretation
ratio = metrics['transient_energy_ratio']
if ratio > 0.7:
    print("🥁 Highly percussive/transient-heavy (drums, percussion)")
elif ratio > 0.5:
    print("🎸 Moderately percussive (rock, energetic music)")
elif ratio > 0.3:
    print("🎵 Balanced mix (pop, balanced production)")
else:
    print("🎹 Harmonic/sustained-heavy (pads, vocals, strings)")

#transient_density = 0.3803  # or metrics['transient_energy_ratio'] use this metric