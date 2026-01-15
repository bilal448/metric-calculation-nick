# Inharmonicity in sound is the deviation of a sound's overtones (partials) from perfect integer multiples of the fundamental frequency,
# creating a richer, sometimes dissonant timbre, prominent in instruments like pianos, bells, and drums due to physical stiffness or shape,
# and it's why pianos need "stretch tuning" for octaves.


import librosa
import numpy as np
from scipy.signal import find_peaks


audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# 1. Find fundamental frequency
f0 = librosa.yin(y, fmin=50, fmax=2000)
f0 = f0[f0 > 0]  # Remove unvoiced
if len(f0) == 0:
    print("No pitch detected")
else:
    f0_mean = np.mean(f0)

    # 2. Get average magnitude spectrum
    stft = librosa.stft(y)
    magnitude = np.mean(np.abs(stft), axis=1)
    freqs = librosa.fft_frequencies(sr=sr)

    # 3. Find spectral peaks (partials)
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
    peak_freqs = freqs[peaks]
    peak_mags = magnitude[peaks]

    # 4. For each harmonic, find deviation
    deviations = []

    for n in range(1, 10):  # Check harmonics 1-9
        ideal_harmonic = n * f0_mean

        # Find peaks near this harmonic (within ±50 Hz)
        nearby = np.abs(peak_freqs - ideal_harmonic) < 50

        if np.any(nearby):
            # Pick the strongest peak nearby
            nearby_peaks = peak_freqs[nearby]
            nearby_mags = peak_mags[nearby]
            actual_freq = nearby_peaks[np.argmax(nearby_mags)]

            # Calculate fractional deviation
            deviation = (actual_freq - ideal_harmonic) / ideal_harmonic
            deviations.append(abs(deviation))

    # 5. Inharmonicity = average deviation
    if deviations:
        inharmonicity = np.mean(deviations)
        print(f"F0: {f0_mean:.1f} Hz")
        print(f"Inharmonicity: {inharmonicity:.4f}")
        print(f"Deviations: {[f'{d:.4f}' for d in deviations[:5]]}")
    else:
        print("No harmonics detected")