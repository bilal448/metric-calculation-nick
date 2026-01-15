# "Warmth" in sound describes a pleasant, rich, full-bodied audio quality, characterized by boosted lower-midrange and
# bass frequencies, smooth highs, and often subtle harmonic distortion, creating an inviting, natural, or "analog" feel,
# contrasting with harsh "cold" sounds

import librosa
import numpy as np

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Get magnitude spectrum
stft = librosa.stft(y)
magnitude = np.abs(stft)
freqs = librosa.fft_frequencies(sr=sr)

# Low frequencies (20-500 Hz = warmth region)
low_mask = (freqs >= 20) & (freqs <= 500)
total_energy = np.sum(magnitude ** 2)
low_energy = np.sum(magnitude[low_mask, :] ** 2)

warmth = low_energy / total_energy

print(f"Warmth (low freq ratio): {warmth:.4f}")