#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_datasets.py — build small Libri2Mix-style 2-speaker evaluation sets
under four acoustic conditions, in the EXACT folder layout this repo consumes.

Four conditions (2x2 robustness grid):
    clean         : s1 + s2
    noise         : s1 + s2 + additive noise
    reverb        : reverb(s1) + reverb(s2)
    noise_reverb  : reverb(s1) + reverb(s2) + additive noise

Per-condition output layout (mirrors LibriMix wav8k/max/<set>/):
    <out>/<condition>/
        mix/<key>.wav     # the mixture handed to the separation model
        s1/<key>.wav      # ground-truth source 1 (post-reverb where applicable)
        s2/<key>.wav      # ground-truth source 2
        text_spk1         # "<key> TRANSCRIPT" lines (for WER)
        text_spk2
        metadata.csv

All audio is mono, 8000 Hz float wav — matching Separation_nets.py's output.

Speech sources: if --source-dir points at a folder of real .wav clips they are
used (paired round-robin); otherwise speech-like signals are synthesised so the
scaffold runs anywhere (CI / no-data smoke tests).
"""

import argparse
import csv
import os

import numpy as np
import soundfile as sf

CONDITIONS = ("clean", "noise", "reverb", "noise_reverb")
SR = 8000


def _rng(seed):
    return np.random.default_rng(seed)


def synth_voice(rng, dur_s, sr=SR):
    """Cheap speech-like signal: a few amplitude-modulated formant tones."""
    t = np.arange(int(dur_s * sr)) / sr
    f0 = rng.uniform(90, 180)                      # pitch
    formants = rng.uniform(300, 3000, size=3)      # formant centres
    sig = np.zeros_like(t)
    for k, f in enumerate(np.r_[f0, formants]):
        sig += (1.0 / (k + 1)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # syllable-rate envelope so PESQ/STOI/VAD see "activity"
    env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 5) * t)) + 0.1
    sig = sig * env
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.7
    return sig.astype("float32")


def load_real_sources(source_dir, sr=SR):
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(source_dir, "*.wav")))
    out = []
    for p in paths:
        x, fsr = sf.read(p)
        if x.ndim > 1:
            x = x.mean(1)
        if fsr != sr:
            import librosa
            x = librosa.resample(x.astype("float32"), orig_sr=fsr, target_sr=sr)
        out.append((os.path.splitext(os.path.basename(p))[0], x.astype("float32")))
    return out


def reverb(sig, rng, sr=SR):
    """Apply a short synthetic exponential-decay room impulse response."""
    rt = rng.uniform(0.2, 0.5)  # decay time (s)
    n = int(0.4 * sr)
    ir = (rng.standard_normal(n) * np.exp(-np.arange(n) / (rt * sr))).astype("float32")
    ir[0] = 1.0  # direct path
    wet = np.convolve(sig, ir)[: len(sig)]
    return (wet / (np.max(np.abs(wet)) + 1e-8) * 0.7).astype("float32")


def mix_to_snr(target, noise, rng, snr_db=None):
    if snr_db is None:
        snr_db = rng.uniform(0, 10)
    tp = np.mean(target ** 2) + 1e-12
    npow = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(tp / (npow * (10 ** (snr_db / 10))))
    return (target + scale * noise).astype("float32")


def build_condition(cond, out_root, n_pairs, sources, rng, dur_s):
    cdir = os.path.join(out_root, cond)
    for sub in ("mix", "s1", "s2"):
        os.makedirs(os.path.join(cdir, sub), exist_ok=True)

    meta_rows = []
    t1_lines, t2_lines = [], []
    count = 0
    for i in range(n_pairs):
        # pick / synthesise the two source signals
        if sources:
            (k1, a1), (k2, a2) = sources[(2 * i) % len(sources)], sources[(2 * i + 1) % len(sources)]
            n = min(len(a1), len(a2))
            a1, a2 = a1[:n].copy(), a2[:n].copy()
            txt1, txt2 = k1, k2
        else:
            a1 = synth_voice(rng, dur_s)
            a2 = synth_voice(rng, dur_s)
            txt1 = "the quick brown fox jumps over the lazy dog"
            txt2 = "she sells sea shells by the sea shore"

        s1, s2 = a1, a2
        if "reverb" in cond:
            s1, s2 = reverb(a1, rng), reverb(a2, rng)

        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        mix = (s1 + s2).astype("float32")
        if "noise" in cond:
            noise = rng.standard_normal(n).astype("float32")
            mix = mix_to_snr(mix, noise, rng)

        peak = np.max(np.abs(mix)) + 1e-8
        mix = (mix / peak * 0.9).astype("float32")

        key = f"{cond}_{i:04d}.wav"
        sf.write(os.path.join(cdir, "mix", key), mix, SR)
        sf.write(os.path.join(cdir, "s1", key), s1, SR)
        sf.write(os.path.join(cdir, "s2", key), s2, SR)
        kk = os.path.splitext(key)[0]
        t1_lines.append(f"{kk} {txt1}")
        t2_lines.append(f"{kk} {txt2}")
        meta_rows.append({"key": key, "condition": cond, "length": n, "sr": SR})
        count += 1

    with open(os.path.join(cdir, "text_spk1"), "w") as f:
        f.write("\n".join(t1_lines) + "\n")
    with open(os.path.join(cdir, "text_spk2"), "w") as f:
        f.write("\n".join(t2_lines) + "\n")
    with open(os.path.join(cdir, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "condition", "length", "sr"])
        w.writeheader()
        w.writerows(meta_rows)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="synth_data", help="Output root directory")
    p.add_argument("--n-pairs", type=int, default=3, help="Mixtures per condition")
    p.add_argument("--dur", type=float, default=3.0, help="Synthetic clip duration (s)")
    p.add_argument("--source-dir", default="", help="Optional folder of real .wav speech clips")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   help="Subset of conditions to build")
    args = p.parse_args()

    rng = _rng(args.seed)
    sources = load_real_sources(args.source_dir) if args.source_dir else []
    src_kind = f"{len(sources)} real clips" if sources else "synthetic speech-like"
    print(f"Building {args.n_pairs} pair(s)/condition from {src_kind} into {args.out}/\n")

    counts = {}
    for cond in args.conditions:
        if cond not in CONDITIONS:
            raise SystemExit(f"Unknown condition: {cond} (choose from {CONDITIONS})")
        counts[cond] = build_condition(cond, args.out, args.n_pairs, sources, rng, args.dur)

    print("Per-condition mixture counts:")
    print("-" * 40)
    for cond in args.conditions:
        print(f"  {cond:14s} {counts[cond]:4d} mixtures")
    print("-" * 40)
    print(f"  {'TOTAL':14s} {sum(counts.values()):4d} mixtures")


if __name__ == "__main__":
    main()
