# Structural novelty of sounds refers to the detection of significant changes or
# transitions in the acoustic properties of an audio signal over time, which often
# mark boundaries between meaningful structural segments

"""
Spectral-Based Structural Novelty
Based on the standard MIR approach using temporal derivatives
"""

import librosa
import numpy as np


def get_structural_novelty_spectral(audio_path):
    """
    Calculate structural novelty using spectral-based method

    Steps:
    1. Time-frequency representation (spectrogram)
    2. Feature compression (log scale)
    3. Temporal derivative (frame-to-frame differences)
    4. Half-wave rectification (keep only positive changes)
    5. Accumulation across frequency bands

    Returns:
        std (float): Standard deviation of novelty function
        boundaries (list): Detected boundary times
        novelty_curve (array): Full novelty function
    """
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # 2. Time-Frequency Representation (Mel Spectrogram)
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)

    # 3. Logarithmic Compression (perceptual scaling)
    Y = librosa.power_to_db(S, ref=np.max)

    # 4. Temporal Derivative (frame-to-frame differences)
    # Compute differences between consecutive frames
    diff = np.diff(Y, axis=1)  # Y(n+1,k) - Y(n,k)

    # 5. Half-Wave Rectification (keep only positive changes)
    diff_positive = np.maximum(0, diff)

    # 6. Accumulation across frequency bands
    # Sum over all frequency bins for each time frame
    novelty_curve = np.sum(diff_positive, axis=0)

    # Normalize
    if np.max(novelty_curve) > 0:
        novelty_curve = novelty_curve / np.max(novelty_curve)

    # 7. Calculate STD
    std = float(np.std(novelty_curve))

    # 8. Peak Detection for boundaries
    # Adaptive threshold
    threshold = np.mean(novelty_curve) + 0.3 * np.std(novelty_curve)

    peaks = librosa.util.peak_pick(
        novelty_curve.astype(np.float64),
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=float(threshold),
        wait=10
    )

    boundaries = librosa.frames_to_time(peaks, sr=sr, hop_length=512).tolist()

    return std, boundaries, novelty_curve


# Alternative: Use librosa's built-in onset_strength (does exactly this!)
def get_structural_novelty_onset(audio_path):
    """
    Using librosa's onset_strength which implements spectral novelty
    This is the standard implementation of the formula you described
    """
    y, sr = librosa.load(audio_path, sr=None)

    # librosa.onset.onset_strength implements:
    # - Mel spectrogram
    # - Temporal differences
    # - Half-wave rectification
    # - Accumulation
    novelty_curve = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # Calculate STD
    std = float(np.std(novelty_curve))

    # Detect peaks
    threshold = np.mean(novelty_curve) + 0.3 * np.std(novelty_curve)

    peaks = librosa.util.peak_pick(
        novelty_curve.astype(np.float64),
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=float(threshold),
        wait=10
    )

    boundaries = librosa.frames_to_time(peaks, sr=sr, hop_length=512).tolist()

    return std, boundaries, novelty_curve


# Complete implementation with all options
def calculate_structural_novelty(audio_path, method='spectral'):
    """
    Calculate structural novelty with multiple methods

    Args:
        audio_path: Path to audio file
        method: 'spectral' (manual), 'onset' (librosa built-in), or 'both'

    Returns:
        dict with std, boundaries, and novelty curve
    """
    if method == 'spectral':
        std, boundaries, curve = get_structural_novelty_spectral(audio_path)
    elif method == 'onset':
        std, boundaries, curve = get_structural_novelty_onset(audio_path)
    else:  # both
        std1, bounds1, curve1 = get_structural_novelty_spectral(audio_path)
        std2, bounds2, curve2 = get_structural_novelty_onset(audio_path)
        return {
            'spectral': {'std': std1, 'boundaries': bounds1, 'curve': curve1},
            'onset': {'std': std2, 'boundaries': bounds2, 'curve': curve2}
        }

    return {
        'std': std,
        'boundaries': boundaries,
        'novelty_curve': curve
    }


# USAGE EXAMPLE
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    print("=" * 70)
    print("SPECTRAL-BASED STRUCTURAL NOVELTY")
    print("=" * 70)

    # Method 1: Manual spectral implementation
    print("\n1. Manual Spectral Method:")
    std1, bounds1, curve1 = get_structural_novelty_spectral(audio_file)
    print(f"   STD: {std1:.6f}")
    print(f"   Boundaries: {bounds1}")

    # Method 2: Using librosa's onset_strength (recommended)
    print("\n2. Librosa onset_strength (Standard MIR approach):")
    std2, bounds2, curve2 = get_structural_novelty_onset(audio_file)
    print(f"   STD: {std2:.6f}")
    print(f"   Boundaries: {bounds2}")

    # Debug info
    print(f"\n   Novelty curve stats:")
    print(f"   Mean: {np.mean(curve2):.4f}")
    print(f"   Max: {np.max(curve2):.4f}")
    print(f"   Min: {np.min(curve2):.4f}")

    # If still no boundaries, try lower threshold
    if len(bounds2) == 0:
        print("\n   No boundaries found. Trying lower threshold...")
        peaks = librosa.util.peak_pick(
            curve2.astype(np.float64),
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=3,
            delta=0.05,  # Very low threshold
            wait=5
        )
        bounds_low = librosa.frames_to_time(peaks, sr=None, hop_length=512).tolist()
        print(f"   Found {len(bounds_low)} boundaries: {bounds_low}")