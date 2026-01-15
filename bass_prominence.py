# Bass prominence in sounds refers to the emphasis on low-frequency tones (typically 20-250 Hz) that provide music
# with depth, rhythm, and emotional impact

"""
Bass Prominence using Demucs
Measures the perceived loudness/emphasis of low-frequency sounds
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_bass_prominence(audio_path):
    """
    Calculate bass prominence using Demucs to separate bass

    Methods:
    1. Bass energy ratio (bass RMS / total RMS)
    2. Low-frequency spectral energy (20-200 Hz vs total)
    3. Bass-to-mid ratio

    Returns:
        bass_prominence (float): Bass prominence score (0-1)
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

    # Extract separated sources
    # Index: 0=drums, 1=bass, 2=other, 3=vocals
    bass = sources[0, 1].cpu().numpy()
    drums = sources[0, 0].cpu().numpy()
    other = sources[0, 2].cpu().numpy()
    vocals = sources[0, 3].cpu().numpy()

    # Convert to mono
    bass_mono = np.mean(bass, axis=0)
    original_mono = np.mean(waveform[0].cpu().numpy(), axis=0)

    # Method 1: RMS Energy Ratio
    bass_rms = np.sqrt(np.mean(bass ** 2))
    drums_rms = np.sqrt(np.mean(drums ** 2))
    other_rms = np.sqrt(np.mean(other ** 2))
    vocals_rms = np.sqrt(np.mean(vocals ** 2))
    total_rms = bass_rms + drums_rms + other_rms + vocals_rms

    bass_energy_ratio = float(bass_rms / (total_rms + 1e-10))

    # Method 2: Spectral Analysis (Low-frequency energy)
    # Analyze 20-200 Hz range
    stft_bass = librosa.stft(bass_mono, n_fft=2048, hop_length=512)
    stft_original = librosa.stft(original_mono, n_fft=2048, hop_length=512)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Low-frequency range (20-200 Hz)
    low_freq_mask = (freqs >= 20) & (freqs <= 200)
    low_freq_energy = np.mean(np.abs(stft_original[low_freq_mask, :]) ** 2)

    # Mid-frequency range (200-2000 Hz)
    mid_freq_mask = (freqs >= 200) & (freqs <= 2000)
    mid_freq_energy = np.mean(np.abs(stft_original[mid_freq_mask, :]) ** 2)

    # Total spectral energy
    total_spectral_energy = np.mean(np.abs(stft_original) ** 2)

    # Bass spectral ratio
    bass_spectral_ratio = float(low_freq_energy / (total_spectral_energy + 1e-10))

    # Bass-to-mid ratio
    bass_to_mid_ratio = float(low_freq_energy / (mid_freq_energy + 1e-10))

    # Combined bass prominence score (weighted average)
    bass_prominence = float(
        0.5 * bass_energy_ratio +
        0.3 * bass_spectral_ratio +
        0.2 * min(bass_to_mid_ratio, 1.0)  # Normalize to max 1.0
    )

    return bass_prominence


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    bass_prominence = get_bass_prominence(audio_file)
    print(f"Bass Prominence: {bass_prominence:.4f}")