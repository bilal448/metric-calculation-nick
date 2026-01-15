# There is no universal "bass percentage" of sound, as it varies greatly depending on the genre of music,
# the specific mix, the audio equipment, and personal preference. Bass refers to low-end frequencies, typically
# between 20 Hz and 250 Hz, which add depth and richness to audio.

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class BassPercentageAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate all stems"""
        print("=" * 70)
        print("BASS PERCENTAGE ANALYZER")
        print("=" * 70)
        print(f"\nLoading: {self.audio_file}")

        # Load audio
        audio, sr = librosa.load(self.audio_file, sr=self.sr, mono=False)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.eval()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Separate sources
        print("Separating stems (vocals, drums, bass, other)...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract stems: drums (0), bass (1), other (2), vocals (3)
        self.stems['drums'] = librosa.to_mono(sources[0].numpy())
        self.stems['bass'] = librosa.to_mono(sources[1].numpy())
        self.stems['other'] = librosa.to_mono(sources[2].numpy())
        self.stems['vocals'] = librosa.to_mono(sources[3].numpy())

        print("✓ Separation complete!\n")

    def calculate_bass_percentage(self):
        """Calculate what percentage of the mix is bass"""
        print("=" * 70)
        print("BASS PERCENTAGE ANALYSIS")
        print("=" * 70)

        # Calculate RMS energy for each stem
        bass_rms = np.sqrt(np.mean(self.stems['bass'] ** 2))
        drums_rms = np.sqrt(np.mean(self.stems['drums'] ** 2))
        vocals_rms = np.sqrt(np.mean(self.stems['vocals'] ** 2))
        other_rms = np.sqrt(np.mean(self.stems['other'] ** 2))

        # Total energy
        total_rms = bass_rms + drums_rms + vocals_rms + other_rms

        if total_rms > 0:
            bass_percentage = (bass_rms / total_rms) * 100
            drums_percentage = (drums_rms / total_rms) * 100
            vocals_percentage = (vocals_rms / total_rms) * 100
            other_percentage = (other_rms / total_rms) * 100
        else:
            bass_percentage = 0
            drums_percentage = 0
            vocals_percentage = 0
            other_percentage = 0

        # Convert to dB
        bass_db = 20 * np.log10(bass_rms + 1e-10)
        drums_db = 20 * np.log10(drums_rms + 1e-10)
        vocals_db = 20 * np.log10(vocals_rms + 1e-10)
        other_db = 20 * np.log10(other_rms + 1e-10)

        print("\n[1] ENERGY DISTRIBUTION")
        print("-" * 70)

        # Visual bars
        print(f"\nBass:    {bass_percentage:>6.2f}%  {'█' * int(bass_percentage / 2)}")
        print(f"Drums:   {drums_percentage:>6.2f}%  {'█' * int(drums_percentage / 2)}")
        print(f"Vocals:  {vocals_percentage:>6.2f}%  {'█' * int(vocals_percentage / 2)}")
        print(f"Other:   {other_percentage:>6.2f}%  {'█' * int(other_percentage / 2)}")

        print(f"\n[2] BASS FOCUS ANALYSIS")
        print("-" * 70)
        print(f"\n{'=' * 30}")
        print(f"BASS PERCENTAGE: {bass_percentage:.2f}%")
        print(f"{'=' * 30}")

        # RMS levels
        print(f"\nRMS Energy Levels:")
        print(f"  Bass:   {bass_rms:.6f} ({bass_db:.1f} dB)")
        print(f"  Drums:  {drums_rms:.6f} ({drums_db:.1f} dB)")
        print(f"  Vocals: {vocals_rms:.6f} ({vocals_db:.1f} dB)")
        print(f"  Other:  {other_rms:.6f} ({other_db:.1f} dB)")

        # Bass contribution analysis
        print(f"\n[3] INTERPRETATION")
        print("-" * 70)

        if bass_percentage < 10:
            print("\n→ MINIMAL BASS")
            print("  Very low bass presence")
            print("  Typical of: Acoustic, a cappella, some classical")
        elif bass_percentage < 20:
            print("\n→ SUBTLE/LIGHT BASS")
            print("  Bass present but understated")
            print("  Typical of: Pop, folk, jazz with upright bass")
        elif bass_percentage < 30:
            print("\n→ BALANCED BASS")
            print("  Bass shares space with other elements")
            print("  Typical of: Rock, indie, standard pop production")
        elif bass_percentage < 40:
            print("\n→ PROMINENT BASS")
            print("  Strong bass presence")
            print("  Typical of: Funk, R&B, modern pop, trap")
        else:
            print("\n→ BASS-DOMINANT MIX")
            print("  Bass is a primary driver of the mix")
            print("  Typical of: Hip-hop, EDM, dubstep, drum & bass, reggae")

        # Low-end analysis (bass + kick drum)
        rhythm_section_rms = bass_rms + drums_rms
        if total_rms > 0:
            low_end_percentage = (rhythm_section_rms / total_rms) * 100

            print(f"\n[4] LOW-END FOUNDATION")
            print("-" * 70)
            print(f"\nLow-End (Bass + Drums): {low_end_percentage:.2f}%")

            if low_end_percentage > 60:
                print("  → VERY STRONG low-end (heavy, powerful)")
            elif low_end_percentage > 45:
                print("  → STRONG low-end (modern production)")
            elif low_end_percentage > 30:
                print("  → MODERATE low-end (balanced)")
            else:
                print("  → LIGHT low-end (mid/high-focused)")

        # Bass-to-drums ratio
        if drums_rms > 0:
            bass_to_drums = bass_rms / drums_rms
            bass_to_drums_db = 20 * np.log10(bass_to_drums)

            print(f"\n[5] BASS-TO-DRUMS BALANCE")
            print("-" * 70)
            print(f"\nBass-to-Drums Ratio: {bass_to_drums:.2f} ({bass_to_drums_db:+.1f} dB)")

            if bass_to_drums_db > 6:
                print("  → Bass MUCH LOUDER than drums (bass-driven)")
            elif bass_to_drums_db > 0:
                print("  → Bass LOUDER than drums (bass emphasis)")
            elif bass_to_drums_db > -6:
                print("  → Bass BALANCED with drums (typical mix)")
            else:
                print("  → Bass QUIETER than drums (drum-focused)")

        # Genre-specific bass frequency analysis
        bass_audio = self.stems['bass']

        # Analyze sub-bass (20-60 Hz) vs bass (60-250 Hz)
        stft = librosa.stft(bass_audio, n_fft=4096)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Sub-bass energy
        sub_bass_mask = (freqs >= 20) & (freqs < 60)
        sub_bass_energy = np.sum(magnitude[sub_bass_mask, :])

        # Mid-bass energy
        mid_bass_mask = (freqs >= 60) & (freqs < 250)
        mid_bass_energy = np.sum(magnitude[mid_bass_mask, :])

        total_bass_energy = sub_bass_energy + mid_bass_energy

        if total_bass_energy > 0:
            sub_bass_percent = (sub_bass_energy / total_bass_energy) * 100
            mid_bass_percent = (mid_bass_energy / total_bass_energy) * 100

            print(f"\n[6] BASS FREQUENCY DISTRIBUTION")
            print("-" * 70)
            print(f"\nSub-Bass (20-60 Hz):   {sub_bass_percent:.1f}%")
            print(f"Mid-Bass (60-250 Hz):  {mid_bass_percent:.1f}%")

            if sub_bass_percent > 40:
                print("  → HIGH sub-bass content (club/EDM style)")
            elif sub_bass_percent > 25:
                print("  → MODERATE sub-bass (modern production)")
            else:
                print("  → LOW sub-bass (traditional/acoustic bass)")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = BassPercentageAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_bass_percentage()

