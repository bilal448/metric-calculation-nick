# Vocal presence refers to how clearly and powerfully a voice cuts through sound, often achieved through
# controlled breath, intentional pitch/pace/volume, and emphasizing mid-range frequencies (2-5 kHz) in audio

"""
Vocal Presence using Demucs - Fixed for Windows
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_vocal_presence(audio_path):
    """
    Calculate vocal presence using Demucs source separation

    Returns:
        vocal_presence (float): Vocal presence score (0-1)
    """
    # Load Demucs model
    model = get_model('htdemucs')
    model.eval()

    # Load audio with librosa instead of torchaudio
    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    # Convert to torch tensor
    if y.ndim == 1:
        # Mono to stereo
        waveform = torch.from_numpy(np.stack([y, y]))
    else:
        # Already stereo
        waveform = torch.from_numpy(y)

    # Ensure float32
    waveform = waveform.float()

    # Add batch dimension
    waveform = waveform.unsqueeze(0)

    # Separate sources
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract separated sources
    # Index: 0=drums, 1=bass, 2=other, 3=vocals
    vocals = sources[0, 3].cpu().numpy()
    drums = sources[0, 0].cpu().numpy()
    bass = sources[0, 1].cpu().numpy()
    other = sources[0, 2].cpu().numpy()

    # Calculate RMS energy
    vocal_rms = np.sqrt(np.mean(vocals ** 2))
    drums_rms = np.sqrt(np.mean(drums ** 2))
    bass_rms = np.sqrt(np.mean(bass ** 2))
    other_rms = np.sqrt(np.mean(other ** 2))

    # Total energy of all sources
    total_rms = vocal_rms + drums_rms + bass_rms + other_rms

    # Vocal presence = vocal energy / total energy
    vocal_presence = float(vocal_rms / (total_rms + 1e-10))

    return vocal_presence


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    presence = get_vocal_presence(audio_file)
    print(f"Vocal Presence: {presence:.4f}")