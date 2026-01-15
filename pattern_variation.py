# "Pattern variation of sounds" primarily refers to the systematic and predictable ways that sound units (phonemes)
# in a language are pronounced differently depending on their context, without changing the word's meaning.
# This phenomenon is studied in phonology, the branch of linguistics concerned with the abstract sound systems
# and patterns of languages.

import librosa
import numpy as np
from scipy.stats import entropy


def get_pattern_variation(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Temporal variation
    temporal_var = np.mean(np.std(np.diff(mfcc, axis=1), axis=1))

    # Pattern entropy
    pattern_ent = np.mean([entropy(np.histogram(mfcc[i], bins=20)[0] + 1e-10)
                           for i in range(mfcc.shape[0])])

    return float(0.6 * temporal_var + 0.4 * pattern_ent)


def get_pattern_variation2(audio_path, sr=22050, hop_length=512):
    """
    Calculate pattern variation in sounds
    Measures how sound patterns vary throughout the audio

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length: Hop length for feature extraction

    Returns:
        pattern_variation (float): Overall pattern variation score
        metrics (dict): Detailed variation metrics
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Extract multiple acoustic features to capture pattern variations

    # 1. MFCCs - captures timbral patterns
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    # 2. Spectral features - captures frequency patterns
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    # 3. Chroma - captures harmonic patterns
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # 4. Zero crossing rate - captures temporal patterns
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    # Calculate variation metrics for each feature

    # Temporal variation (frame-to-frame changes)
    mfcc_variation = np.mean(np.std(np.diff(mfcc, axis=1), axis=1))
    chroma_variation = np.mean(np.std(np.diff(chroma, axis=1), axis=1))
    spectral_centroid_variation = np.std(np.diff(spectral_centroid))
    spectral_contrast_variation = np.mean(np.std(np.diff(spectral_contrast, axis=1), axis=1))
    zcr_variation = np.std(np.diff(zcr))

    # Pattern diversity (entropy of distributions)
    mfcc_entropy = np.mean([entropy(np.histogram(mfcc[i], bins=20)[0] + 1e-10) for i in range(mfcc.shape[0])])
    chroma_entropy = np.mean([entropy(np.histogram(chroma[i], bins=20)[0] + 1e-10) for i in range(chroma.shape[0])])

    # Statistical variation
    mfcc_std = np.mean(np.std(mfcc, axis=1))
    chroma_std = np.mean(np.std(chroma, axis=1))
    spectral_centroid_std = np.std(spectral_centroid)

    # Coefficient of variation (normalized variability)
    mfcc_cv = np.mean(np.std(mfcc, axis=1) / (np.abs(np.mean(mfcc, axis=1)) + 1e-10))

    # Overall pattern variation score (weighted combination)
    pattern_variation = float(
        0.3 * mfcc_variation +
        0.2 * chroma_variation +
        0.2 * spectral_contrast_variation +
        0.15 * mfcc_entropy +
        0.15 * chroma_entropy
    )

    # Detailed metrics
    metrics = {
        'pattern_variation_score': float(pattern_variation),
        'temporal_variations': {
            'mfcc_variation': float(mfcc_variation),
            'chroma_variation': float(chroma_variation),
            'spectral_centroid_variation': float(spectral_centroid_variation),
            'spectral_contrast_variation': float(spectral_contrast_variation),
            'zcr_variation': float(zcr_variation)
        },
        'pattern_diversity': {
            'mfcc_entropy': float(mfcc_entropy),
            'chroma_entropy': float(chroma_entropy)
        },
        'statistical_variation': {
            'mfcc_std': float(mfcc_std),
            'chroma_std': float(chroma_std),
            'spectral_centroid_std': float(spectral_centroid_std),
            'mfcc_coefficient_variation': float(mfcc_cv)
        }
    }

    return pattern_variation, metrics
# Usage
variation = get_pattern_variation("trimmed_output33.wav")
print(f"Pattern Variation: {variation:.6f}")

# Method 1: Full calculation
print("\n1. FULL PATTERN VARIATION:")
variation, metrics = get_pattern_variation2("trimmed_output33.wav")
print(f"Pattern Variation Score: {variation:.6f}")
print(f"\nTemporal Variations:")
for key, val in metrics['temporal_variations'].items():
    print(f"  {key}: {val:.6f}")
print(f"\nPattern Diversity:")
for key, val in metrics['pattern_diversity'].items():
    print(f"  {key}: {val:.6f}")
print(f"\nStatistical Variation:")
for key, val in metrics['statistical_variation'].items():
    print(f"  {key}: {val:.6f}")