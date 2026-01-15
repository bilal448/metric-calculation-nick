# The concept of "hook repetition offsets of sound" refers to a production technique that uses variations in
# timing, placement, and arrangement each time a short, catchy musical phrase (a hook or motif) is repeated to
# maintain listener interest and prevent monotony.


import numpy as np
import librosa
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import warnings

warnings.filterwarnings('ignore')


class HookRepetitionAnalyzer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.sr = 22050
        self.y = None
        self.results = {}

    def run_analysis(self):
        """Main execution flow - returns the results dictionary"""
        self.load_audio()
        self.compute_features()
        self.compute_similarity()
        self.detect_hooks()
        return self.finalize_results()

    def load_audio(self):
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        self.results['duration'] = float(librosa.get_duration(y=self.y, sr=self.sr))

    def compute_features(self):
        # Using a balanced stack of Chroma (Harmony) and MFCC (Timbre)
        hop = 512
        chroma = librosa.util.normalize(librosa.feature.chroma_cqt(y=self.y, sr=self.sr, hop_length=hop))
        mfcc = librosa.util.normalize(librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13, hop_length=hop))

        # We combine them into one representation of the "sound" at each moment
        self.combined_features = np.vstack([chroma * 1.5, mfcc[:5]])

    def compute_similarity(self):
        # Calculate how similar every frame is to every other frame
        S = 1 - cdist(self.combined_features.T, self.combined_features.T, metric='cosine')
        self.similarity_matrix = median_filter(S, size=3)

    def detect_hooks(self, min_seg_len=4, threshold=0.7):
        n_frames = self.similarity_matrix.shape[0]
        repetitions = []

        # Scans for diagonal lines (repeated sequences)
        for offset in range(min_seg_len, n_frames - min_seg_len):
            diag = np.diagonal(self.similarity_matrix, offset=offset)
            if len(diag) < 10: continue

            diag_smooth = np.convolve(diag, np.ones(5) / 5, mode='same')
            peaks, _ = find_peaks(diag_smooth, height=threshold, distance=min_seg_len)

            for peak in peaks:
                time_start = librosa.frames_to_time(peak, sr=self.sr)
                time_rep = librosa.frames_to_time(peak + offset, sr=self.sr)

                repetitions.append({
                    'time_1': float(time_start),
                    'time_2': float(time_rep),
                    'offset': float(time_rep - time_start),
                    'similarity': float(diag_smooth[peak])
                })

        # Filter out duplicates and group them
        repetitions.sort(key=lambda x: x['similarity'], reverse=True)
        unique_hooks = []
        for rep in repetitions:
            if not any(abs(rep['time_1'] - u['time_1']) < 2.0 for u in unique_hooks):
                unique_hooks.append(rep)

        self.results['raw_repetitions'] = unique_hooks

    def finalize_results(self):
        """Processes the math into human-readable metrics and prints them"""
        hooks = self.results.get('raw_repetitions', [])

        # Calculate 'Stickiness' (Catchiness)
        # Based on how many repetitions exist and how strong the similarity is
        if not hooks:
            return {"status": "No hooks detected"}

        total_sim = sum([h['similarity'] for h in hooks])
        avg_stickiness = total_sim / len(hooks) if hooks else 0

        final_output = {
            "duration_sec": self.results['duration'],
            "total_hook_count": len(hooks),
            "avg_stickiness_score": round(float(avg_stickiness), 3),
            "hook_offsets": [
                {"start": h['time_1'], "repeat_at": h['time_2'], "delay": h['offset']}
                for h in hooks[:5]  # Top 5 hooks
            ]
        }

        # Print to Console
        print("\n" + "=" * 40)
        print("   HOOK REPETITION RESULTS")
        print("=" * 40)
        print(f"Total Repetitions Found: {final_output['total_hook_count']}")
        print(f"Average Catchiness:      {final_output['avg_stickiness_score']}")
        print("-" * 40)
        for i, h in enumerate(final_output['hook_offsets']):
            print(
                f"Hook {i + 1}: Origin {h['start']:.2f}s -> Repeats at {h['repeat_at']:.2f}s (Gap: {h['delay']:.2f}s)")
        print("=" * 40 + "\n")

        return final_output


# Usage
if __name__ == "__main__":
    # Ensure you have your wav file path here
    # If using Demucs, point this to your 'vocals.wav' for best results!
    analyzer = HookRepetitionAnalyzer("trimmed_output33.wav")
    results = analyzer.run_analysis()