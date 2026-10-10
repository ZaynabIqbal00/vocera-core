"""
Vocera - evaluation v2 (needs vocera_compare.py in the same folder).

What it adds over v1:
  1. Uses ALL clip pairs (not 90 random ones), so the error rate is far more reliable.
  2. ENROLLMENT AVERAGING: a speaker is enrolled with the average voiceprint of several
     clips (like the real system), then tested with single clips they were NOT enrolled with.
  3. Breaks single-clip results down by phrase (imposter saying the same phrase vs other phrase).
  4. Optional silence trimming and a minimum clip length.
  5. Shows which speakers cause the most errors.

USAGE (from D:\\vocera-core\\eval, venv active):
  python vocera_eval_v2.py ..\\data\\raw_15_speakers
  python vocera_eval_v2.py ..\\data\\raw_15_speakers --trim
  python vocera_eval_v2.py ..\\data\\raw_15_speakers --trim --n-enroll 8 --min-dur 1.0
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vocera_compare import Campp, Ecapa, collect_speakers, to_16k_mono


# ----------------------------------------------------------------- models
class CamppEmb(Campp):
    """CAM++ that can also return the voiceprint itself (not only a pair score)."""

    def __init__(self, model_id=None):
        super().__init__(model_id)
        self.cache = {}

    def _embedding(self, path):
        if path not in self.cache:
            try:
                with torch.no_grad():
                    emb = self.pipe.forward(self.pipe.preprocess([path]))[0]
            except Exception:
                out = self.pipe([path, path], output_emb=True)
                emb = torch.from_numpy(np.asarray(out["embs"][0]))
            self.cache[path] = emb
        return self.cache[path]


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


# ------------------------------------------------------------------ audio
def trim_silence(audio, sr, top_db=35, frame_ms=20):
    """Cut silence at the start and end of a clip."""
    n = int(sr * frame_ms / 1000)
    frames = len(audio) // n
    if frames < 3:
        return audio
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms / rms.max())
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return audio
    start = max(keep[0] - 1, 0) * n
    end = min(keep[-1] + 2, frames) * n
    out = audio[start:end]
    return out if len(out) >= 0.3 * sr else audio


def prepare(path, trim, top_db):
    p = to_16k_mono(path)
    if not trim:
        return p
    out = Path(p).with_name(f"{Path(p).stem}_trim{top_db}.wav")
    if not out.exists():
        audio, sr = sf.read(p, dtype="float32")
        sf.write(str(out), trim_silence(audio, sr, top_db), sr)
    return str(out)


# ---------------------------------------------------------------- metrics
def eer(same, diff):
    """Equal error rate and the score threshold where it happens."""
    same = np.sort(np.asarray(same))
    diff = np.sort(np.asarray(diff))
    th = np.unique(np.concatenate([same, diff]))
    frr = np.searchsorted(same, th, side="left") / len(same)        # real person rejected
    far = 1 - np.searchsorted(diff, th, side="left") / len(diff)    # stranger accepted
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(th[i])


def single_clip_eval(E, spk, ph, use_phrase):
    """Every pair of clips, one clip against one clip."""
    S = E @ E.T
    iu = np.triu_indices(len(spk), 1)
    sc = S[iu]
    same = spk[iu[0]] == spk[iu[1]]
    rows = {"all pairs": eer(sc[same], sc[~same]) + (int(same.sum()), int((~same).sum()))}
    if use_phrase:
        samep = ph[iu[0]] == ph[iu[1]]
        for label, mask in (("imposter says a DIFFERENT phrase", ~same & ~samep),
                            ("imposter says the SAME phrase", ~same & samep)):
            if mask.sum() > 0:
                rows[label] = eer(sc[same], sc[mask]) + (int(same.sum()), int(mask.sum()))
    return rows


def enrollment_eval(E, spk, n_enroll, repeats, seed=0):
    """Enroll with the average of n_enroll clips, test with single clips not used for enrollment."""
    rng = np.random.default_rng(seed)
    names = np.unique(spk)
    eers, gen_all, gen_spk, imp_all = [], [], [], []
    for _ in range(repeats):
        cents, test_mask = [], np.ones(len(spk), bool)
        for s in names:
            idx = np.where(spk == s)[0]
            chosen = rng.choice(idx, size=min(n_enroll, len(idx) - 1), replace=False)
            test_mask[chosen] = False
            cents.append(unit(E[chosen].mean(axis=0)))
        T = np.where(test_mask)[0]
        scores = E[T] @ np.stack(cents).T
        own = spk[T][:, None] == names[None, :]
        g, im = scores[own], scores[~own]
        eers.append(eer(g, im))
        gen_all.append(g)
        gen_spk.append(np.repeat(spk[T], 1))
        imp_all.append(im)
    return eers, np.concatenate(gen_all), np.concatenate(gen_spk), np.concatenate(imp_all)


# ------------------------------------------------------------------- main
def evaluate(E, spk, ph, model_name, args, out_rows):
    use_phrase = len(set(ph)) < len(ph) * 0.5
    print(f"\n==================== {model_name} ====================")
    print("A) Single clip vs single clip (no enrollment averaging)")
    for label, (e, t, ns, nd) in single_clip_eval(E, spk, ph, use_phrase).items():
        print(f"   {label:34s} EER {e*100:5.1f}%   threshold {t:.3f}   ({ns} same / {nd} different pairs)")
        out_rows.append([model_name, "single clip: " + label, round(e, 4), round(t, 4), ns, nd])

    eers, gen, gen_spk, imp = enrollment_eval(E, spk, args.n_enroll, args.repeats)
    vals = np.array([x[0] for x in eers]) * 100
    pooled_e, pooled_t = eer(gen, imp)
    print(f"\nB) Enrolled with the average of {args.n_enroll} clips, tested with single clips (repeated {args.repeats}x)")
    print(f"   EER {vals.mean():.1f}% on average (best run {vals.min():.1f}%, worst run {vals.max():.1f}%)")
    print(f"   threshold about {pooled_t:.3f}   ({len(gen)} genuine / {len(imp)} imposter trials in total)")
    out_rows.append([model_name, f"enrolled with {args.n_enroll} clips", round(pooled_e, 4), round(pooled_t, 4), len(gen), len(imp)])

    print("\n   Speakers with the most rejected genuine clips (at that threshold):")
    worst = []
    for s in np.unique(gen_spk):
        g = gen[gen_spk == s]
        worst.append((float(np.mean(g < pooled_t)), float(g.mean()), s))
    for frr, mean_s, s in sorted(worst, reverse=True)[:5]:
        print(f"     {s:12s} rejected {frr*100:4.0f}% of the time, average score {mean_s:.3f}")


def main():
    p = argparse.ArgumentParser(description="Vocera evaluation v2")
    p.add_argument("root", help="folder with one sub-folder per speaker")
    p.add_argument("--n-enroll", type=int, default=5, help="clips averaged to enroll a speaker")
    p.add_argument("--repeats", type=int, default=10, help="random enrollment choices to average over")
    p.add_argument("--trim", action="store_true", help="cut silence at clip start and end")
    p.add_argument("--top-db", type=int, default=35, help="silence level for --trim")
    p.add_argument("--min-dur", type=float, default=0.0, help="ignore clips shorter than this (seconds)")
    p.add_argument("--out", default="v2_results.csv")
    p.add_argument("--no-ecapa", action="store_true")
    p.add_argument("--no-campp", action="store_true")
    args = p.parse_args()

    items = []  # (speaker, phrase, prepared path)
    for name, files in collect_speakers(args.root).items():
        for f in files:
            items.append((name, f.stem.lower(), prepare(f, args.trim, args.top_db)))
    durs = np.array([sf.info(i[2]).duration for i in items])
    keep = durs >= args.min_dur
    items = [i for i, k in zip(items, keep) if k]
    durs = durs[keep]
    spk = np.array([i[0] for i in items])
    ph = np.array([i[1] for i in items])
    print(f"{len(set(spk))} speakers, {len(items)} clips, clip length avg {durs.mean():.1f}s "
          f"(min {durs.min():.1f}s, max {durs.max():.1f}s), {len(set(ph))} distinct file names"
          f"{'  [silence trimmed]' if args.trim else ''}")
    if len(set(ph)) >= len(ph) * 0.5:
        print("Note: file names are mostly unique, so the same-phrase breakdown is skipped.")

    models = []
    for cls, skip in ((Ecapa, args.no_ecapa), (CamppEmb, args.no_campp)):
        if not skip:
            try:
                models.append(cls())
            except Exception as e:
                print(f"[!] Could not load {cls.__name__}: {e}")
    if not models:
        raise SystemExit("No model loaded. Send the error above.")

    rows = []
    for m in models:
        E = np.stack([unit(m._embedding(i[2]).detach().cpu().numpy()) for i in items])
        evaluate(E, spk, ph, m.name, args, rows)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "eer", "threshold", "n_same_or_genuine", "n_different_or_imposter"])
        w.writerows(rows)
    print(f"\nSummary saved to {args.out}")


if __name__ == "__main__":
    main()
