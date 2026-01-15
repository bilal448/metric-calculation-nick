# Mel-Frequency Cepstral Coefficients (MFCCs) are features extracted from audio to create a compact, human-perception-based
# representation of a sound, commonly used in speech and speaker recognition by modeling the vocal tract's frequency response
# on a non-linear Mel scale, mimicking how humans hear pitch and loudness

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Extract MFCCs (default: 20 coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# mfccs shape: (13, num_frames)
# Get mean of each coefficient
mfcc_mean = np.mean(mfccs, axis=1)

print(f"MFCC shape: {mfccs.shape}")
print(f"MFCC means: {mfcc_mean}")