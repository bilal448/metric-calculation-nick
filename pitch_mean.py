# The pitch of a sound is the subjective perception of how high or low a sound is, and it is primarily determined by the physical
# property of the sound wave's frequency.

"""
Pitch Detection using Demucs - Extract vocals first, then detect pitch
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_pitch_with_demucs(audio_path):
    """
    Calculate pitch using Demucs to separate vocals first

    Returns:
        pitch_mean (float): Mean pitch in Hz
        pitch_std (float): Standard deviation of pitch
        voiced_frames (int): Number of frames with detected pitch
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

    # Detect pitch using pYIN on separated vocals
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
        pitch_mean = float(np.mean(confident_voiced))
        pitch_std = float(np.std(confident_voiced))
        voiced_frames = len(confident_voiced)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        voiced_frames = 0

    return pitch_mean, pitch_std, voiced_frames


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    pitch_mean, pitch_std, voiced_frames = get_pitch_with_demucs(audio_file)

    if voiced_frames > 0:
        print(f"Mean Pitch: {pitch_mean:.2f} Hz")
        print(f"Pitch Std Dev: {pitch_std:.2f} Hz")
        print(f"Voiced Frames: {voiced_frames}")
    else:
        print("No pitch detected")