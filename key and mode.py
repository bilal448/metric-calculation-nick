# The "key of sound" in music refers to the central note (tonic) and the specific set of notes (scale, major or minor)
# that a piece of music is built around, creating a sense of "home" or stability, with most melodies and harmonies revolving
# around these notes, like C Major or A Minor (Mode Is calculated here as well)

# Mode in music refers to whether a piece is in Major or Minor:
#
# Major: Sounds happy, bright, uplifting
# Minor: Sounds sad, dark, somber

import librosa
import numpy as np

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Krumhansl-Schmuckler key profiles
major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Extract chroma
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_mean = np.mean(chroma, axis=1)

# Correlate with each key
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
correlations = []

for i in range(12):
    # Rotate profile to match different keys
    rotated_major = np.roll(major_profile, i)
    rotated_minor = np.roll(minor_profile, i)

    # Correlation
    corr_major = np.corrcoef(chroma_mean, rotated_major)[0, 1]
    corr_minor = np.corrcoef(chroma_mean, rotated_minor)[0, 1]

    correlations.append((i, 'Major', corr_major))
    correlations.append((i, 'Minor', corr_minor))

# Find best match
best = max(correlations, key=lambda x: x[2])
key_note = notes[best[0]]
mode = best[1]

print(f"Detected Key: {key_note} {mode}")
print(f"Confidence: {best[2]:.3f}")