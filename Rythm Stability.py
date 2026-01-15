# Rhythm stability of sound refers to the consistency and predictability of a rhythmic pattern over time,
# which is crucial for human perception, musical cohesion, and neurological processing.
# A stable rhythm provides an underlying pulse or "beat" that the brain can easily track and synchronize with.


import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Detect beats
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Calculate inter-beat intervals (IBI)
if len(beat_times) > 1:
    inter_beat_intervals = np.diff(beat_times)

    # Rhythm stability metrics
    ibi_mean = np.mean(inter_beat_intervals)
    ibi_std = np.std(inter_beat_intervals)

    # Coefficient of Variation (CV) - standard metric for stability
    # Lower = more stable
    rhythm_stability_cv = ibi_std / ibi_mean if ibi_mean > 0 else 0

    # Rhythm stability as 0-1 score (higher = more stable)
    rhythm_stability = 1.0 / (1.0 + rhythm_stability_cv)

    print(f"Mean Beat Interval: {ibi_mean:.4f} seconds")
    print(f"Std Dev of Beat Intervals: {ibi_std:.4f} seconds")
    print(f"Coefficient of Variation: {rhythm_stability_cv:.4f}")
    print(f"Rhythm Stability (0-1): {rhythm_stability:.4f}")
else:
    print("Not enough beats detected to calculate rhythm stability")