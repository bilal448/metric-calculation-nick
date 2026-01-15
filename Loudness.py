import librosa
import numpy as np

# Load the audio file
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate

############### Loudness  #####################

# Apply A-weighting (perceptual frequency weighting)
y_weighted = librosa.perceptual_weighting(
    librosa.stft(y),
    frequencies=librosa.fft_frequencies(sr=sr),
    kind='A'  # A-weighting
)

# Convert back and compute RMS
y_weighted_time = librosa.istft(y_weighted)
rms_weighted = np.sqrt(np.mean(y_weighted_time**2))

# Convert to dB
loudness_db = 20 * np.log10(rms_weighted + 1e-10)

print(f"A-weighted Loudness: {loudness_db:.2f} dB")

# Or frame-by-frame loudness
rms_weighted_frames = librosa.feature.rms(y=y_weighted_time)
loudness_frames_db = librosa.amplitude_to_db(rms_weighted_frames, ref=1.0)

print(f"Loudness over time (shape): {loudness_frames_db.shape}")
print(f"Mean loudness: {np.mean(loudness_frames_db):.2f} dB")

# RMS energy (simple loudness approximation)
rms = np.sqrt(np.mean(y**2))
loudness_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)

print(f"RMS Loudness: {loudness_db:.2f} dB")

# Or using librosa's built-in
rms_frames = librosa.feature.rms(y=y)[0]
loudness_db_frames = librosa.amplitude_to_db(rms_frames, ref=1.0)
mean_loudness = np.mean(loudness_db_frames)

print(f"Mean Loudness (dB): {mean_loudness:.2f} dB")


#A weighting is considered more accurate