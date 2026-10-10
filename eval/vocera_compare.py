"""
Vocera - first experiment: do same-person pairs score higher than different-person pairs?

Compares two speaker models on your own recordings:
  - ECAPA-TDNN (SpeechBrain, baseline)
  - CAM++      (3D-Speaker via ModelScope, main model)

USAGE
  python vocera_compare.py pair a.wav b.wav
  python vocera_compare.py folder data/speakers

Scores are cosine similarities. Higher = more likely the same person.
The decision threshold is NOT fixed here: it must be tuned later on your own data.
"""
import argparse
import csv
import hashlib
import itertools
import random
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
CACHE = Path(tempfile.gettempdir()) / "vocera_16k"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ---------------------------------------------------------------- audio prep
def to_16k_mono(path):
    """Return a path to a 16 kHz mono wav copy of `path` (both models expect this)."""
    path = Path(path)
    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:10]
    out = CACHE / f"{path.stem}_{key}.wav"
    if out.exists():
        return str(out)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype("float32")
    sf.write(str(out), audio, TARGET_SR)
    return str(out)


# -------------------------------------------------------------------- models
class Ecapa:
    name = "ECAPA-TDNN"

    def __init__(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older SpeechBrain versions
            from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        self.cache = {}

    def _embedding(self, path):
        # Audio is loaded with soundfile (already 16 kHz mono) to avoid the
        # torchaudio/torchcodec error on new versions.
        import torch

        if path not in self.cache:
            audio, _ = sf.read(path, dtype="float32")
            signal = torch.from_numpy(audio).unsqueeze(0)
            with torch.no_grad():
                self.cache[path] = self.model.encode_batch(signal).squeeze()
        return self.cache[path]

    def score(self, a, b):
        import torch.nn.functional as F

        return float(F.cosine_similarity(self._embedding(a), self._embedding(b), dim=0))


class Campp:
    name = "CAM++"

    def __init__(self, model_id=None):
        from modelscope.pipelines import pipeline

        kwargs = {}
        if model_id is None:  # the version that already worked on your laptop
            model_id = "damo/speech_campplus_sv_en_voxceleb_16k"
            kwargs["model_revision"] = "v1.0.2"
        self.pipe = pipeline(task="speaker-verification", model=model_id, **kwargs)

    def score(self, a, b):
        result = self.pipe([a, b])
        if "score" not in result:
            raise RuntimeError(f"Unexpected CAM++ output: {result}")
        return float(result["score"])


def load_models(use_ecapa=True, use_campp=True, campp_id=None):
    models = []
    if use_ecapa:
        try:
            models.append(Ecapa())
        except Exception as e:
            print(f"[!] Could not load ECAPA-TDNN: {e}")
    if use_campp:
        try:
            models.append(Campp(campp_id) if campp_id else Campp())
        except Exception as e:
            print(f"[!] Could not load CAM++: {e}")
    if not models:
        raise SystemExit("No model could be loaded. Send the error messages above and we will fix it.")
    return models


# ------------------------------------------------------------------- metrics
def equal_error_rate(same_scores, diff_scores):
    """Rough EER: threshold where false accepts ~ false rejects."""
    same = np.asarray(same_scores)
    diff = np.asarray(diff_scores)
    best_gap, best = 2.0, (None, None)
    for t in np.unique(np.concatenate([same, diff])):
        frr = float(np.mean(same < t))   # real customer rejected
        far = float(np.mean(diff >= t))  # stranger accepted
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, best = gap, ((frr + far) / 2, float(t))
    return best  # (eer, threshold)


# --------------------------------------------------------------------- modes
def run_pair(models, file_a, file_b):
    a, b = to_16k_mono(file_a), to_16k_mono(file_b)
    print(f"\nA: {file_a}\nB: {file_b}\n")
    for m in models:
        print(f"{m.name:12s} score = {m.score(a, b):.4f}")
    print("\nHigher score = more likely the same person.")


def collect_speakers(root):
    speakers = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXT)
            if files:
                speakers[d.name] = files
    return speakers


def build_pairs(speakers, max_same_per_speaker=6, seed=0):
    rng = random.Random(seed)
    same = []
    for spk, files in speakers.items():
        combos = list(itertools.combinations(files, 2))
        rng.shuffle(combos)
        same += [(a, b, 1) for a, b in combos[:max_same_per_speaker]]
    names = list(speakers)
    diff = []
    if len(names) >= 2:
        while len(diff) < len(same):
            s1, s2 = rng.sample(names, 2)
            diff.append((rng.choice(speakers[s1]), rng.choice(speakers[s2]), 0))
    return same + diff


def run_folder(models, root, out_csv="eval/results_quick.csv", max_same=6):
    speakers = collect_speakers(root)
    print(f"Found {len(speakers)} speakers, {sum(len(v) for v in speakers.values())} files.")
    if len(speakers) < 2:
        raise SystemExit("Need at least 2 speaker folders with audio files.")
    pairs = build_pairs(speakers, max_same)
    print(f"Testing {len(pairs)} pairs ({len(pairs)//2} same-person, {len(pairs)//2} different-person)\n")

    rows, scores = [], {m.name: {1: [], 0: []} for m in models}
    for i, (a, b, label) in enumerate(pairs, 1):
        pa, pb = to_16k_mono(a), to_16k_mono(b)
        row = {"file_a": str(a), "file_b": str(b), "same_person": label}
        for m in models:
            s = m.score(pa, pb)
            row[m.name] = round(s, 5)
            scores[m.name][label].append(s)
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(pairs)} pairs done")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== RESULTS (higher score = more similar) ===")
    for m in models:
        same, diff = scores[m.name][1], scores[m.name][0]
        eer, thr = equal_error_rate(same, diff)
        print(f"\n{m.name}")
        print(f"  mean score, SAME person      : {np.mean(same):.3f}")
        print(f"  mean score, DIFFERENT people : {np.mean(diff):.3f}")
        print(f"  rough EER                    : {eer*100:.1f}%  (at score {thr:.3f})")
    print(f"\nAll scores saved to {out_csv}")
    print("Good sign: SAME is clearly higher than DIFFERENT. With ~15 speakers the EER is only a rough guide.")


# ----------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="Vocera speaker model comparison")
    sub = p.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("pair", help="compare two audio files")
    pp.add_argument("file_a")
    pp.add_argument("file_b")

    pf = sub.add_parser("folder", help="test all speakers in a folder")
    pf.add_argument("root")
    pf.add_argument("--max-same", type=int, default=6, help="max same-person pairs per speaker")
    pf.add_argument("--out", default="eval/results_quick.csv")

    for sp in (pp, pf):
        sp.add_argument("--no-ecapa", action="store_true")
        sp.add_argument("--no-campp", action="store_true")
        sp.add_argument("--campp-id", default=None, help="other CAM++ ModelScope id to try")

    args = p.parse_args()
    models = load_models(not args.no_ecapa, not args.no_campp, args.campp_id)
    if args.mode == "pair":
        run_pair(models, args.file_a, args.file_b)
    else:
        run_folder(models, args.root, args.out, args.max_same)


if __name__ == "__main__":
    main()