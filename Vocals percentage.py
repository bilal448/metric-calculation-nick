# The "percentage" of a mix that vocals occupy is not a fixed number but depends entirely on the genre,
# artistic intent, and balance within the mix. There are no strict rules, but rather general guidelines
# for vocal levels relative to other elements in a song.

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class VocalsPercentageAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate all stems"""
        print("=" * 70)
        print("VOCALS PERCENTAGE ANALYZER")
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

    def calculate_vocals_percentage(self):
        """Calculate what percentage of the mix is vocals"""
        print("=" * 70)
        print("VOCALS PERCENTAGE ANALYSIS")
        print("=" * 70)

        # Calculate RMS energy for each stem
        vocals_rms = np.sqrt(np.mean(self.stems['vocals'] ** 2))
        drums_rms = np.sqrt(np.mean(self.stems['drums'] ** 2))
        bass_rms = np.sqrt(np.mean(self.stems['bass'] ** 2))
        other_rms = np.sqrt(np.mean(self.stems['other'] ** 2))

        # Total energy
        total_rms = vocals_rms + drums_rms + bass_rms + other_rms

        if total_rms > 0:
            vocals_percentage = (vocals_rms / total_rms) * 100
            drums_percentage = (drums_rms / total_rms) * 100
            bass_percentage = (bass_rms / total_rms) * 100
            other_percentage = (other_rms / total_rms) * 100
        else:
            vocals_percentage = 0
            drums_percentage = 0
            bass_percentage = 0
            other_percentage = 0

        # Convert to dB for reference
        vocals_db = 20 * np.log10(vocals_rms + 1e-10)
        drums_db = 20 * np.log10(drums_rms + 1e-10)
        bass_db = 20 * np.log10(bass_rms + 1e-10)
        other_db = 20 * np.log10(other_rms + 1e-10)

        print("\n[1] ENERGY DISTRIBUTION")
        print("-" * 70)

        # Visual bars
        print(f"\nVocals:  {vocals_percentage:>6.2f}%  {'█' * int(vocals_percentage / 2)}")
        print(f"Drums:   {drums_percentage:>6.2f}%  {'█' * int(drums_percentage / 2)}")
        print(f"Bass:    {bass_percentage:>6.2f}%  {'█' * int(bass_percentage / 2)}")
        print(f"Other:   {other_percentage:>6.2f}%  {'█' * int(other_percentage / 2)}")

        print(f"\n[2] VOCALS FOCUS ANALYSIS")
        print("-" * 70)
        print(f"\n{'=' * 30}")
        print(f"VOCALS PERCENTAGE: {vocals_percentage:.2f}%")
        print(f"{'=' * 30}")

        # RMS levels
        print(f"\nRMS Energy Levels:")
        print(f"  Vocals: {vocals_rms:.6f} ({vocals_db:.1f} dB)")
        print(f"  Drums:  {drums_rms:.6f} ({drums_db:.1f} dB)")
        print(f"  Bass:   {bass_rms:.6f} ({bass_db:.1f} dB)")
        print(f"  Other:  {other_rms:.6f} ({other_db:.1f} dB)")

        # Genre classification based on vocals percentage
        print(f"\n[3] INTERPRETATION")
        print("-" * 70)

        if vocals_percentage < 15:
            print("\n→ INSTRUMENTAL/MINIMAL VOCALS")
            print("  Very low vocal presence")
            print("  Typical of: Instrumental tracks, EDM, ambient")
        elif vocals_percentage < 25:
            print("\n→ BACKGROUND/SUPPORTING VOCALS")
            print("  Vocals present but not dominant")
            print("  Typical of: Some EDM, instrumental-focused pop")
        elif vocals_percentage < 35:
            print("\n→ BALANCED MIX")
            print("  Vocals share space with instruments")
            print("  Typical of: Rock, indie, alternative")
        elif vocals_percentage < 45:
            print("\n→ VOCAL-FORWARD MIX")
            print("  Strong vocal presence")
            print("  Typical of: Pop, R&B, country")
        else:
            print("\n→ VOCAL-DOMINANT MIX")
            print("  Vocals are the primary focus")
            print("  Typical of: Ballads, acoustic, a cappella")

        # Vocal prominence ratio
        backing_track_rms = drums_rms + bass_rms + other_rms
        if backing_track_rms > 0:
            vocal_prominence = vocals_rms / backing_track_rms
            vocal_prominence_db = 20 * np.log10(vocal_prominence)

            print(f"\n[4] VOCAL PROMINENCE")
            print("-" * 70)
            print(f"\nVocals-to-Backing Ratio: {vocal_prominence:.2f} ({vocal_prominence_db:+.1f} dB)")

            if vocal_prominence_db > 6:
                print("  → Vocals MUCH LOUDER than backing (ballad style)")
            elif vocal_prominence_db > 0:
                print("  → Vocals LOUDER than backing (vocal-forward)")
            elif vocal_prominence_db > -6:
                print("  → Vocals BALANCED with backing (typical mix)")
            else:
                print("  → Vocals QUIETER than backing (instrumental focus)")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = VocalsPercentageAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_vocals_percentage()

