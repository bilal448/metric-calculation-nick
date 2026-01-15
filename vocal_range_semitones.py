"""
Vocal ranges are measured in semitones (half steps) by noting the lowest to highest pitches a singer can comfortably hit,
typically spanning 2-3 octaves, with female voices (Soprano, Mezzo, Alto) generally higher than male voices (Tenor, Baritone, Bass),
though ranges overlap, like a Soprano's C4-C6 vs. a Tenor's C3-C5, showing a difference of about 24 semitones (two full octaves)
between their standard starting notes. For instance, a Soprano might cover C4 to C6, while a Tenor covers C3 to C5,
meaning the Soprano starts higher and goes higher, covering a wider span in semitones.

"""

"""
Vocal Range in Semitones using Demucs
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_vocal_range_semitones(audio_path):
    """
    Calculate vocal range in semitones using Demucs

    Formula: ST = 12 * log2(f_max / f_min)

    Returns:
        vocal_range_semitones (float): Vocal range in semitones
        min_pitch_hz (float): Lowest pitch in Hz
        max_pitch_hz (float): Highest pitch in Hz
    """
    # Load Demucs model
    model = get_model('htdemucs')
    model.eval()

    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    # Convert to torch tensor
    if y.ndim == 1:
        waveform = torch.from_numpy(np.stack([y, y]))
    else:
        waveform = torch.from_numpy(y)

    waveform = waveform.float().unsqueeze(0)

    # Separate sources with Demucs
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract vocals (index 3)
    vocals = sources[0, 3].cpu().numpy()

    # Convert to mono
    vocals_mono = np.mean(vocals, axis=0)

    # Detect pitch using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        vocals_mono,
        sr=sr,
        fmin=50,
        fmax=2000
    )

    # Filter by confidence threshold
    confident_voiced = f0[voiced_probs > 0.5]
    confident_voiced = confident_voiced[~np.isnan(confident_voiced)]

    if len(confident_voiced) > 0:
        min_pitch_hz = float(np.min(confident_voiced))
        max_pitch_hz = float(np.max(confident_voiced))

        # Calculate semitones: ST = 12 * log2(f_max / f_min)
        vocal_range_semitones = float(12 * np.log2(max_pitch_hz / min_pitch_hz))
    else:
        min_pitch_hz = 0.0
        max_pitch_hz = 0.0
        vocal_range_semitones = 0.0

    return vocal_range_semitones, min_pitch_hz, max_pitch_hz


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    semitones, min_hz, max_hz = get_vocal_range_semitones(audio_file)

    print(f"vocal_range_semitones: {semitones:.2f}")
    print(f"min_pitch_hz: {min_hz:.2f}")
    print(f"max_pitch_hz: {max_hz:.2f}")

    # Interpretation
    octaves = semitones / 12
    print(f"\nVocal Range: {octaves:.2f} octaves")