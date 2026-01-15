# The question of what "percentage" a drum contributes to the overall sound isn't a fixed scientific number,
# but rather a principle in acoustics and drumming that suggests drum heads and their tuning are responsible for
# approximately 80-90% of a drum's sound

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class DrumsPercentageAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate all stems"""
        print("=" * 70)
        print("DRUMS PERCENTAGE ANALYZER")
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

    def calculate_drums_percentage(self):
        """Calculate what percentage of the mix is drums"""
        print("=" * 70)
        print("DRUMS PERCENTAGE ANALYSIS")
        print("=" * 70)

        # Calculate RMS energy for each stem
        drums_rms = np.sqrt(np.mean(self.stems['drums'] ** 2))
        vocals_rms = np.sqrt(np.mean(self.stems['vocals'] ** 2))
        bass_rms = np.sqrt(np.mean(self.stems['bass'] ** 2))
        other_rms = np.sqrt(np.mean(self.stems['other'] ** 2))

        # Total energy
        total_rms = drums_rms + vocals_rms + bass_rms + other_rms

        if total_rms > 0:
            drums_percentage = (drums_rms / total_rms) * 100
            vocals_percentage = (vocals_rms / total_rms) * 100
            bass_percentage = (bass_rms / total_rms) * 100
            other_percentage = (other_rms / total_rms) * 100
        else:
            drums_percentage = 0
            vocals_percentage = 0
            bass_percentage = 0
            other_percentage = 0

        # Convert to dB
        drums_db = 20 * np.log10(drums_rms + 1e-10)
        vocals_db = 20 * np.log10(vocals_rms + 1e-10)
        bass_db = 20 * np.log10(bass_rms + 1e-10)
        other_db = 20 * np.log10(other_rms + 1e-10)

        print("\n[1] ENERGY DISTRIBUTION")
        print("-" * 70)

        # Visual bars
        print(f"\nDrums:   {drums_percentage:>6.2f}%  {'█' * int(drums_percentage / 2)}")
        print(f"Vocals:  {vocals_percentage:>6.2f}%  {'█' * int(vocals_percentage / 2)}")
        print(f"Bass:    {bass_percentage:>6.2f}%  {'█' * int(bass_percentage / 2)}")
        print(f"Other:   {other_percentage:>6.2f}%  {'█' * int(other_percentage / 2)}")

        print(f"\n[2] DRUMS FOCUS ANALYSIS")
        print("-" * 70)
        print(f"\n{'=' * 30}")
        print(f"DRUMS PERCENTAGE: {drums_percentage:.2f}%")
        print(f"{'=' * 30}")

        # RMS levels
        print(f"\nRMS Energy Levels:")
        print(f"  Drums:  {drums_rms:.6f} ({drums_db:.1f} dB)")
        print(f"  Vocals: {vocals_rms:.6f} ({vocals_db:.1f} dB)")
        print(f"  Bass:   {bass_rms:.6f} ({bass_db:.1f} dB)")
        print(f"  Other:  {other_rms:.6f} ({other_db:.1f} dB)")

        # Drums contribution analysis
        print(f"\n[3] INTERPRETATION")
        print("-" * 70)

        if drums_percentage < 15:
            print("\n→ MINIMAL DRUMS")
            print("  Very low drum presence")
            print("  Typical of: Ballads, acoustic, ambient, some electronic")
        elif drums_percentage < 25:
            print("\n→ SUBTLE/LIGHT DRUMS")
            print("  Drums present but understated")
            print("  Typical of: Jazz, folk, soft pop")
        elif drums_percentage < 35:
            print("\n→ BALANCED DRUMS")
            print("  Drums share space with other elements")
            print("  Typical of: Pop, indie, R&B")
        elif drums_percentage < 45:
            print("\n→ PROMINENT DRUMS")
            print("  Strong drum presence")
            print("  Typical of: Rock, funk, hip-hop")
        else:
            print("\n→ DRUM-DOMINANT MIX")
            print("  Drums are the primary driver")
            print("  Typical of: Drum & bass, EDM, metal, percussion-heavy tracks")

        # Drums-to-melody ratio
        melodic_rms = vocals_rms + other_rms
        if melodic_rms > 0:
            drums_to_melody = drums_rms / melodic_rms
            drums_to_melody_db = 20 * np.log10(drums_to_melody)

            print(f"\n[4] DRUMS-TO-MELODY RATIO")
            print("-" * 70)
            print(f"\nDrums-to-Melody Ratio: {drums_to_melody:.2f} ({drums_to_melody_db:+.1f} dB)")

            if drums_to_melody_db > 6:
                print("  → Drums MUCH LOUDER than melody (percussion showcase)")
            elif drums_to_melody_db > 0:
                print("  → Drums LOUDER than melody (rhythm-focused)")
            elif drums_to_melody_db > -6:
                print("  → Drums BALANCED with melody (typical mix)")
            else:
                print("  → Drums QUIETER than melody (melody-focused)")

        # Rhythm section analysis (drums + bass)
        rhythm_section_percentage = drums_percentage + bass_percentage

        print(f"\n[5] RHYTHM SECTION STRENGTH")
        print("-" * 70)
        print(f"\nRhythm Section (Drums + Bass): {rhythm_section_percentage:.2f}%")

        if rhythm_section_percentage > 60:
            print("  → VERY STRONG rhythm section (groove-driven)")
        elif rhythm_section_percentage > 45:
            print("  → STRONG rhythm section (typical modern production)")
        elif rhythm_section_percentage > 30:
            print("  → MODERATE rhythm section (balanced)")
        else:
            print("  → LIGHT rhythm section (melody/vocal-focused)")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = DrumsPercentageAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_drums_percentage()

