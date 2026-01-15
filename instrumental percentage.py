# There is no universal or standard "instrumental percentage of sound" for music. The balance
# between instruments (and vocals, if present) is entirely dependent on the specific song, musical
# genre, artistic intent, and personal taste.

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class InstrumentalPercentageAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate all stems"""
        print("=" * 70)
        print("INSTRUMENTAL/OTHER PERCENTAGE ANALYZER")
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

    def calculate_instrumental_percentage(self):
        """Calculate what percentage of the mix is other instruments"""
        print("=" * 70)
        print("INSTRUMENTAL/OTHER PERCENTAGE ANALYSIS")
        print("=" * 70)

        # Calculate RMS energy for each stem
        other_rms = np.sqrt(np.mean(self.stems['other'] ** 2))
        drums_rms = np.sqrt(np.mean(self.stems['drums'] ** 2))
        bass_rms = np.sqrt(np.mean(self.stems['bass'] ** 2))
        vocals_rms = np.sqrt(np.mean(self.stems['vocals'] ** 2))

        # Total energy
        total_rms = other_rms + drums_rms + bass_rms + vocals_rms

        if total_rms > 0:
            other_percentage = (other_rms / total_rms) * 100
            drums_percentage = (drums_rms / total_rms) * 100
            bass_percentage = (bass_rms / total_rms) * 100
            vocals_percentage = (vocals_rms / total_rms) * 100
        else:
            other_percentage = 0
            drums_percentage = 0
            bass_percentage = 0
            vocals_percentage = 0

        # Convert to dB
        other_db = 20 * np.log10(other_rms + 1e-10)
        drums_db = 20 * np.log10(drums_rms + 1e-10)
        bass_db = 20 * np.log10(bass_rms + 1e-10)
        vocals_db = 20 * np.log10(vocals_rms + 1e-10)

        print("\n[1] ENERGY DISTRIBUTION")
        print("-" * 70)

        # Visual bars
        print(f"\nOther:   {other_percentage:>6.2f}%  {'█' * int(other_percentage / 2)}")
        print(f"Drums:   {drums_percentage:>6.2f}%  {'█' * int(drums_percentage / 2)}")
        print(f"Bass:    {bass_percentage:>6.2f}%  {'█' * int(bass_percentage / 2)}")
        print(f"Vocals:  {vocals_percentage:>6.2f}%  {'█' * int(vocals_percentage / 2)}")

        print(f"\n[2] INSTRUMENTAL FOCUS ANALYSIS")
        print("-" * 70)
        print(f"\n{'=' * 30}")
        print(f"OTHER/INSTRUMENTAL: {other_percentage:.2f}%")
        print(f"{'=' * 30}")

        print("\nNote: 'Other' includes guitars, keyboards, synths, strings,")
        print("      brass, woodwinds, and all melodic/harmonic instruments")

        # RMS levels
        print(f"\nRMS Energy Levels:")
        print(f"  Other:  {other_rms:.6f} ({other_db:.1f} dB)")
        print(f"  Drums:  {drums_rms:.6f} ({drums_db:.1f} dB)")
        print(f"  Bass:   {bass_rms:.6f} ({bass_db:.1f} dB)")
        print(f"  Vocals: {vocals_rms:.6f} ({vocals_db:.1f} dB)")

        # Interpretation
        print(f"\n[3] INTERPRETATION")
        print("-" * 70)

        if other_percentage < 15:
            print("\n→ MINIMAL INSTRUMENTATION")
            print("  Very sparse arrangement")
            print("  Typical of: A cappella, minimal electronic, drum-focused")
        elif other_percentage < 25:
            print("\n→ SPARSE INSTRUMENTATION")
            print("  Simple, stripped-down arrangement")
            print("  Typical of: Acoustic, some hip-hop, minimalist production")
        elif other_percentage < 35:
            print("\n→ BALANCED INSTRUMENTATION")
            print("  Moderate instrumental presence")
            print("  Typical of: Pop, indie, singer-songwriter")
        elif other_percentage < 45:
            print("\n→ RICH INSTRUMENTATION")
            print("  Strong instrumental content")
            print("  Typical of: Rock, jazz, full-band production")
        else:
            print("\n→ INSTRUMENT-DOMINANT MIX")
            print("  Instrumental elements are the primary focus")
            print("  Typical of: Orchestral, prog rock, instrumental tracks")

        # Total instrumental content (other + drums + bass)
        total_instrumental_percentage = other_percentage + drums_percentage + bass_percentage

        print(f"\n[4] TOTAL INSTRUMENTAL CONTENT")
        print("-" * 70)
        print(f"\nAll Instruments (Other + Drums + Bass): {total_instrumental_percentage:.2f}%")
        print(f"Vocals: {vocals_percentage:.2f}%")

        if total_instrumental_percentage > 85:
            print("\n  → INSTRUMENTAL-DOMINATED (>85% instruments)")
        elif total_instrumental_percentage > 70:
            print("\n  → INSTRUMENT-HEAVY (70-85% instruments)")
        elif total_instrumental_percentage > 50:
            print("\n  → BALANCED (50-70% instruments)")
        else:
            print("\n  → VOCAL-DOMINATED (<50% instruments)")

        # Instrumentation style
        if vocals_rms > 0:
            instruments_to_vocals = (other_rms + drums_rms + bass_rms) / vocals_rms
            instruments_to_vocals_db = 20 * np.log10(instruments_to_vocals)

            print(f"\n[5] INSTRUMENTATION-TO-VOCALS RATIO")
            print("-" * 70)
            print(f"\nInstruments-to-Vocals: {instruments_to_vocals:.2f} ({instruments_to_vocals_db:+.1f} dB)")

            if instruments_to_vocals_db > 12:
                print("  → HEAVILY instrumental (vocals are sparse/background)")
            elif instruments_to_vocals_db > 6:
                print("  → INSTRUMENT-FORWARD (instruments dominate)")
            elif instruments_to_vocals_db > 0:
                print("  → SLIGHTLY instrument-forward (typical rock/pop)")
            elif instruments_to_vocals_db > -6:
                print("  → BALANCED (equal emphasis)")
            else:
                print("  → VOCAL-FORWARD (instruments support vocals)")

        # Analyze harmonic vs rhythmic balance
        harmonic_rms = other_rms + vocals_rms
        rhythmic_rms = drums_rms + bass_rms

        if harmonic_rms + rhythmic_rms > 0:
            harmonic_percentage = (harmonic_rms / (harmonic_rms + rhythmic_rms)) * 100
            rhythmic_percentage = (rhythmic_rms / (harmonic_rms + rhythmic_rms)) * 100

            print(f"\n[6] HARMONIC VS RHYTHMIC BALANCE")
            print("-" * 70)
            print(f"\nHarmonic (Other + Vocals):  {harmonic_percentage:.1f}%")
            print(f"Rhythmic (Drums + Bass):    {rhythmic_percentage:.1f}%")

            if rhythmic_percentage > 60:
                print("  → RHYTHM-DRIVEN (groove-focused)")
            elif rhythmic_percentage > 40:
                print("  → BALANCED (typical mix)")
            else:
                print("  → MELODY-DRIVEN (harmonic-focused)")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = InstrumentalPercentageAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_instrumental_percentage()
