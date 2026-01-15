# Sub-bass refers to the lowest layer of audio frequencies, typically ranging from 20 Hz to 60 Hz.
# Its presence in sound is often felt as a physical vibration (like a rumble in the chest) more than
# it is consciously heard, adding depth, power, and a foundational "weight" to music and sound design.

"""
Sub-Bass Presence using Demucs
Measures the presence of ultra-low frequencies (20-60 Hz)
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_sub_bass_presence(audio_path):
    """
    Calculate sub-bass presence using Demucs to separate bass

    Sub-bass range: 20-60 Hz

    Returns:
        sub_bass_presence (float): Sub-bass presence score (0-1)
    """
    # Load Demucs model
    model = get_model('htdemucs')
    model.eval()

    # Load audio
    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    if y.ndim == 1:
        waveform = torch.from_numpy(np.stack([y, y]))
    else:
        waveform = torch.from_numpy(y)

    waveform = waveform.float().unsqueeze(0)

    # Separate sources with Demucs
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract bass track (index 1)
    bass = sources[0, 1].cpu().numpy()
    bass_mono = np.mean(bass, axis=0)

    # Original audio mono
    original_mono = np.mean(waveform[0].cpu().numpy(), axis=0)

    # Spectral analysis
    stft_bass = librosa.stft(bass_mono, n_fft=4096, hop_length=512)
    stft_original = librosa.stft(original_mono, n_fft=4096, hop_length=512)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # Sub-bass range (20-60 Hz)
    sub_bass_mask = (freqs >= 20) & (freqs <= 60)
    sub_bass_energy = np.mean(np.abs(stft_bass[sub_bass_mask, :]) ** 2)

    # Bass range (60-200 Hz) for comparison
    bass_mask = (freqs >= 60) & (freqs <= 200)
    bass_energy = np.mean(np.abs(stft_bass[bass_mask, :]) ** 2)

    # Total energy
    total_energy = np.mean(np.abs(stft_original) ** 2)

    # Sub-bass presence as ratio of sub-bass energy to total
    sub_bass_ratio = float(sub_bass_energy / (total_energy + 1e-10))

    # Normalize to 0-1 range (typical sub-bass ratio is 0.001-0.05)
    sub_bass_presence = float(min(sub_bass_ratio * 20, 1.0))

    return sub_bass_presence


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    sub_bass_presence = get_sub_bass_presence(audio_file)
    print(f"Sub-Bass Presence: {sub_bass_presence:.4f}")