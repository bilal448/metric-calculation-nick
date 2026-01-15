import librosa
import numpy as np

#the difference between the quietest and loudest sounds in an audio signal, measured in decibels (dB),
# Load the audio file
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate

# Method 1: Peak-to-RMS ratio (most common)
rms = librosa.feature.rms(y=y)[0]
peak = np.max(np.abs(y))
rms_overall = np.sqrt(np.mean(y**2))

dynamic_range_ratio = peak / rms_overall
dynamic_range_db = 20 * np.log10(peak / rms_overall)

print(f"Dynamic Range (ratio): {dynamic_range_ratio}")
print(f"Dynamic Range (dB): {dynamic_range_db}")

# Method 2: Loudest to quietest RMS frames
rms_max = np.max(rms)
rms_min = np.min(rms[rms > 0])  # Exclude silence
dynamic_range_frames_db = 20 * np.log10(rms_max / rms_min)

print(f"Dynamic Range (frame-based, dB): {dynamic_range_frames_db}")

# Method 3: Using amplitude_to_db for cleaner dB calculation
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
dynamic_range_db_clean = np.max(rms_db) - np.min(rms_db[rms_db > -np.inf])

print(f"Dynamic Range (dB, clean): {dynamic_range_db_clean}")

#Dynamic Range (dB) this is what to use