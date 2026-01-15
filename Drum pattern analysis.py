# Drum pattern analysis of sound involves breaking down rhythms into core components like kick, snare, and hi-hats,
# analyzing their placement in time (meter, groove), instrumentation (timbre, dynamics), and function (pulse, forward motion)
# to understand how they build a track's foundation, express form, and create feel, using techniques from basic grid-based
# observation to advanced digital signal processing for sound classification and rhythm modification


# What specific aspects of the drums do you want to analyze? (e.g., tempo/BPM, onset timing, rhythm patterns, specific drum hits like kick/snare/hi-hat, groove complexity, etc.)


import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class DrumRhythmAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.drums = None
        self.results = {}

    def separate_drums(self):
        """Use Demucs to separate drum track"""
        print(f"Loading audio: {self.audio_file}")

        # Load audio
        audio, sr = librosa.load(self.audio_file, sr=self.sr, mono=False)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        # Load Demucs model
        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.eval()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Separate sources
        print("Separating drums...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract drums (index 1 in htdemucs: drums, bass, other, vocals)
        drums_stereo = sources[1].numpy()

        # Convert to mono for analysis
        self.drums = librosa.to_mono(drums_stereo)
        print("Drum separation complete!")

    def analyze_rhythm_patterns(self):
        """Comprehensive rhythm pattern analysis"""
        print("\n" + "=" * 60)
        print("RHYTHM PATTERN ANALYSIS")
        print("=" * 60)

        hop_length = 512

        # 1. ONSET DETECTION - Find all drum hits
        print("\n[1] Detecting drum onsets...")
        onset_env = librosa.onset.onset_strength(y=self.drums, sr=self.sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onsets, sr=self.sr, hop_length=hop_length)

        self.results['onset_count'] = len(onsets)
        self.results['onset_times'] = onset_times
        print(f"   Found {len(onsets)} drum hits")

        # 2. TEMPO & BEAT TRACKING
        print("\n[2] Analyzing tempo and beats...")
        tempo_array, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length)
        tempo = float(tempo_array[0]) if np.ndim(tempo_array) > 0 else float(tempo_array)
        beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=hop_length)

        self.results['tempo'] = tempo
        self.results['beat_count'] = len(beats)
        self.results['beat_times'] = beat_times
        print(f"   Tempo: {tempo:.1f} BPM")
        print(f"   Beats detected: {len(beats)}")

        # 3. ONSET REGULARITY (Micro-level pattern consistency)
        print("\n[3] Calculating onset regularity...")
        if len(onset_times) >= 2:
            onset_intervals = np.diff(onset_times)
            onset_mean = np.mean(onset_intervals)
            onset_std = np.std(onset_intervals)
            onset_cv = onset_std / onset_mean if onset_mean > 0 else 0
            onset_regularity_score = max(0.0, 1.0 - onset_cv)

            self.results['onset_interval_mean'] = float(onset_mean)
            self.results['onset_interval_std'] = float(onset_std)
            self.results['onset_regularity'] = float(onset_regularity_score)

            print(f"   Average hit spacing: {onset_mean * 1000:.1f} ms")
            print(f"   Spacing variation: {onset_std * 1000:.1f} ms")
            print(f"   Regularity score: {onset_regularity_score:.3f} (0=chaotic, 1=metronomic)")
        else:
            self.results['onset_regularity'] = 0.0

        # 4. BEAT REGULARITY (Macro-level tempo consistency)
        print("\n[4] Calculating beat regularity...")
        if len(beat_times) >= 2:
            beat_intervals = np.diff(beat_times)
            beat_std = np.std(beat_intervals)
            beat_regularity = 1.0 / (1.0 + beat_std)

            self.results['beat_interval_std'] = float(beat_std)
            self.results['beat_regularity'] = float(beat_regularity)

            print(f"   Beat timing variation: {beat_std * 1000:.1f} ms")
            print(f"   Beat regularity: {beat_regularity:.3f} (higher = more steady)")
        else:
            self.results['beat_regularity'] = 0.0

        # 5. RHYTHM COMPLEXITY - Hits per beat
        print("\n[5] Analyzing rhythm complexity...")
        if len(beats) >= 2 and len(onsets) > 0:
            hits_per_beat = len(onsets) / len(beats)
            self.results['hits_per_beat'] = float(hits_per_beat)
            print(f"   Average hits per beat: {hits_per_beat:.2f}")

            # Categorize complexity
            if hits_per_beat < 2:
                complexity = "Simple (sparse pattern)"
            elif hits_per_beat < 4:
                complexity = "Moderate (standard rock/pop)"
            elif hits_per_beat < 6:
                complexity = "Complex (busy pattern)"
            else:
                complexity = "Very Complex (dense/fast fills)"

            self.results['complexity_category'] = complexity
            print(f"   Complexity: {complexity}")

        # 6. SYNCOPATION DETECTION
        print("\n[6] Detecting syncopation...")
        if len(beats) >= 2 and len(onsets) > 0:
            # Check how many onsets fall off-beat
            on_beat_threshold = 0.1  # 100ms tolerance
            off_beat_count = 0

            for onset_time in onset_times:
                # Find closest beat
                distances = np.abs(beat_times - onset_time)
                min_distance = np.min(distances)

                if min_distance > on_beat_threshold:
                    off_beat_count += 1

            syncopation_ratio = off_beat_count / len(onset_times)
            self.results['syncopation_ratio'] = float(syncopation_ratio)
            self.results['off_beat_hits'] = off_beat_count

            print(f"   Off-beat hits: {off_beat_count}/{len(onset_times)}")
            print(f"   Syncopation ratio: {syncopation_ratio:.2f} (0=on-beat, 1=fully syncopated)")

        # 7. GROOVE ANALYSIS - Inter-onset interval distribution
        print("\n[7] Analyzing groove characteristics...")
        if len(onset_times) >= 3:
            onset_intervals = np.diff(onset_times)

            # Find most common interval (the groove)
            hist, bin_edges = np.histogram(onset_intervals, bins=20)
            most_common_idx = np.argmax(hist)
            groove_interval = (bin_edges[most_common_idx] + bin_edges[most_common_idx + 1]) / 2

            self.results['groove_interval'] = float(groove_interval)
            self.results['groove_bpm'] = 60.0 / groove_interval if groove_interval > 0 else 0

            print(f"   Primary groove interval: {groove_interval * 1000:.1f} ms")
            print(f"   Groove BPM equivalent: {60.0 / groove_interval:.1f}")

        # 8. PATTERN DENSITY (hits per second)
        print("\n[8] Calculating pattern density...")
        duration = len(self.drums) / self.sr
        density = len(onset_times) / duration
        self.results['duration'] = float(duration)
        self.results['pattern_density'] = float(density)
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Pattern density: {density:.2f} hits/second")

    def generate_summary(self):
        """Generate human-readable summary"""
        print("\n" + "=" * 60)
        print("RHYTHM PATTERN SUMMARY")
        print("=" * 60)

        print(f"\nBasic Metrics:")
        print(f"  • Tempo: {self.results.get('tempo', 0):.1f} BPM")
        print(f"  • Total drum hits: {self.results.get('onset_count', 0)}")
        print(f"  • Pattern density: {self.results.get('pattern_density', 0):.2f} hits/sec")

        print(f"\nRhythm Characteristics:")
        print(f"  • Onset regularity: {self.results.get('onset_regularity', 0):.3f}")
        print(f"  • Beat steadiness: {self.results.get('beat_regularity', 0):.3f}")
        print(f"  • Complexity: {self.results.get('complexity_category', 'Unknown')}")
        print(f"  • Syncopation: {self.results.get('syncopation_ratio', 0):.2f}")

        print(f"\nInterpretation:")

        # Onset regularity interpretation
        reg = self.results.get('onset_regularity', 0)
        if reg > 0.8:
            print("  • Very tight, metronomic playing (likely programmed/quantized)")
        elif reg > 0.6:
            print("  • Consistent timing with slight human feel")
        elif reg > 0.4:
            print("  • Moderate variation, natural groove")
        else:
            print("  • High variation, loose/swung feel or complex patterns")

        # Syncopation interpretation
        sync = self.results.get('syncopation_ratio', 0)
        if sync > 0.5:
            print("  • Highly syncopated (funk, jazz, or complex rhythms)")
        elif sync > 0.3:
            print("  • Moderate syncopation (contemporary pop/rock)")
        else:
            print("  • Mostly on-beat playing (straightforward patterns)")

        print("\n" + "=" * 60)

    def save_results(self, output_file='drum_analysis_results.txt'):
        """Save results to file"""
        with open(output_file, 'w') as f:
            f.write("DRUM RHYTHM PATTERN ANALYSIS RESULTS\n")
            f.write("=" * 60 + "\n\n")

            for key, value in self.results.items():
                if isinstance(value, (list, np.ndarray)):
                    f.write(f"{key}: [array with {len(value)} elements]\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "trimmed_output33.wav"  # Change this to your WAV file name

    print("=" * 60)
    print("DRUM RHYTHM PATTERN ANALYZER")
    print("=" * 60)

    # Initialize analyzer
    analyzer = DrumRhythmAnalyzer(AUDIO_FILE)

    # Step 1: Separate drums using Demucs
    analyzer.separate_drums()

    # Step 2: Analyze rhythm patterns
    analyzer.analyze_rhythm_patterns()

    # Step 3: Generate summary
    analyzer.generate_summary()

    # Step 4: Save results
    analyzer.save_results()

    print("\nAnalysis complete!")

    #Let's save beat_regularity
