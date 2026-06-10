#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_datasets.py — build single-variable 2-speaker evaluation test sets
from LibriSpeech test-clean, in the folder layout this repo consumes.

Four conditions, each sweeping exactly ONE variable while the others are held
fixed (0 dB SIR, no noise, full overlap, unconstrained gender unless that is
the swept variable):

    sir     : speaker power ratio 10*log10(P_s1/P_s2) in {-5, -2.5, 0, 2.5, 5} dB
    noise   : additive white-noise SNR vs the speech mixture in
              {clean, 15, 10, 5, 0} dB ("clean" adds no noise)
    overlap : overlap ratio (overlapping samples / shorter-source length) in
              {0.0, 0.25, 0.5, 0.75, 1.0}
    gender  : speaker-gender pairing in {same, diff} per SPEAKERS.TXT

Output layout:
    <out>/<condition>/<level>/
        mix/<utt_id>.wav    # the mixture handed to the separation model
        s1/<utt_id>.wav     # ground-truth source 1, as placed in the mix
        s2/<utt_id>.wav     # ground-truth source 2
    <out>/<condition>/manifest.csv
        utt_id,condition,level,mix,ref_1,ref_2,text_1,text_2,seed,src_1,src_2

All audio is mono 8000 Hz float32 wav. Sources are LibriSpeech test-clean
flacs (16 kHz) resampled to 8 kHz; transcripts come from the .trans.txt files.

Every mixture is reproducible from its manifest row alone: --regen re-runs
generate_one() with the recorded src paths / condition / level / seed and
writes a bit-identical mix (see --regen / --regen-out).
"""

import argparse
import csv
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 8000
MAX_SRC_S = 6.0          # cap each source clip (s) to keep sets small
MIN_UTT_S, MAX_UTT_S = 3.0, 10.0   # usable utterance durations in the pool

LEVELS = {
    "sir": ["-5", "-2.5", "0", "2.5", "5"],
    "noise": ["clean", "15", "10", "5", "0"],
    "overlap": ["0.0", "0.25", "0.5", "0.75", "1.0"],
    "gender": ["same", "diff"],
}
CONDITIONS = tuple(LEVELS)


def synth_voice(rng, dur_s, sr=SR):
    """Cheap speech-like signal (kept for run_inference.sh `fixture`)."""
    t = np.arange(int(dur_s * sr)) / sr
    f0 = rng.uniform(90, 180)
    formants = rng.uniform(300, 3000, size=3)
    sig = np.zeros_like(t)
    for k, f in enumerate(np.r_[f0, formants]):
        sig += (1.0 / (k + 1)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 5) * t)) + 0.1
    sig = sig * env
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.7
    return sig.astype("float32")


# ----------------------------------------------------------------------------
# LibriSpeech test-clean pool
# ----------------------------------------------------------------------------
def load_speaker_genders(test_clean_dir):
    """Parse SPEAKERS.TXT (one level above test-clean/) -> {spk_id: 'M'|'F'}."""
    spk_txt = os.path.join(os.path.dirname(os.path.abspath(test_clean_dir)),
                           "SPEAKERS.TXT")
    genders = {}
    with open(spk_txt, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[2] == "test-clean":
                genders[parts[0]] = parts[1]
    return genders


def scan_pool(test_clean_dir, genders):
    """Walk test-clean -> {spk: [(utt_id, flac_path, transcript), ...]}."""
    pool = {}
    for spk in sorted(os.listdir(test_clean_dir)):
        spk_dir = os.path.join(test_clean_dir, spk)
        if not os.path.isdir(spk_dir) or spk not in genders:
            continue
        for chap in sorted(os.listdir(spk_dir)):
            cdir = os.path.join(spk_dir, chap)
            trans = os.path.join(cdir, f"{spk}-{chap}.trans.txt")
            if not os.path.isfile(trans):
                continue
            texts = {}
            with open(trans, encoding="utf-8") as f:
                for line in f:
                    uid, _, txt = line.strip().partition(" ")
                    texts[uid] = txt
            for uid, txt in texts.items():
                p = os.path.join(cdir, uid + ".flac")
                if not os.path.isfile(p) or not txt:
                    continue
                info = sf.info(p)
                dur = info.frames / info.samplerate
                if MIN_UTT_S <= dur <= MAX_UTT_S:
                    pool.setdefault(spk, []).append((uid, p, txt))
    return pool


def load_audio_8k(path):
    x, fsr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    if fsr != SR:
        assert fsr % SR == 0, f"unsupported sample rate {fsr} for {path}"
        x = resample_poly(x, 1, fsr // SR)
    return x[: int(MAX_SRC_S * SR)]


# ----------------------------------------------------------------------------
# Mixture generation — deterministic given (condition, level, src paths, seed)
# ----------------------------------------------------------------------------
def generate_one(condition, level, src_1, src_2, seed):
    """Return (mix, ref_1, ref_2) float32 arrays for one manifest row."""
    rng = np.random.default_rng(int(seed))
    a1 = load_audio_8k(src_1)
    a2 = load_audio_8k(src_2)

    if condition == "overlap":
        # 0 dB SIR on the full clips, then place s2 so the overlapping region
        # is exactly round(level * len(shorter)) samples.
        a2 = a2 * np.sqrt(np.mean(a1 ** 2) / (np.mean(a2 ** 2) + 1e-12))
        ratio = float(level)
        n1, n2 = len(a1), len(a2)
        ov = int(round(ratio * min(n1, n2)))
        start2 = n1 - ov
        total = max(n1, start2 + n2)
        ref1 = np.zeros(total); ref1[:n1] = a1
        ref2 = np.zeros(total); ref2[start2:start2 + n2] = a2
    else:
        n = min(len(a1), len(a2))
        a1, a2 = a1[:n], a2[:n]
        sir_db = float(level) if condition == "sir" else 0.0
        a2 = a2 * np.sqrt(np.mean(a1 ** 2) /
                          (np.mean(a2 ** 2) + 1e-12) / 10 ** (sir_db / 10))
        ref1, ref2 = a1, a2

    mix = ref1 + ref2
    if condition == "noise" and level != "clean":
        snr_db = float(level)
        noise = rng.standard_normal(len(mix))
        noise *= np.sqrt(np.mean(mix ** 2) /
                         (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
        mix = mix + noise

    # one shared gain so SIR / SNR / overlap relations are untouched
    peak = np.max(np.abs(mix))
    g = 0.9 / peak if peak > 0.9 else 1.0
    return ((mix * g).astype("float32"),
            (ref1 * g).astype("float32"),
            (ref2 * g).astype("float32"))


def pick_pair(condition, level, pool, genders, rng):
    """Pick (utt_id, path, text) for two distinct speakers per the condition."""
    spks = sorted(pool)
    if condition == "gender":
        males = [s for s in spks if genders[s] == "M"]
        females = [s for s in spks if genders[s] == "F"]
        if level == "same":
            grp = males if rng.integers(2) else females
            s1, s2 = rng.choice(grp, size=2, replace=False)
        else:
            s1, s2 = rng.choice(males), rng.choice(females)
            if rng.integers(2):
                s1, s2 = s2, s1
    else:
        s1, s2 = rng.choice(spks, size=2, replace=False)
    u1 = pool[str(s1)][rng.integers(len(pool[str(s1)]))]
    u2 = pool[str(s2)][rng.integers(len(pool[str(s2)]))]
    return str(s1), u1, str(s2), u2


def build_condition(condition, out_root, n_pairs, pool, genders, rng):
    cdir = os.path.join(out_root, condition)
    rows = []
    for level in LEVELS[condition]:
        ldir = os.path.join(cdir, level)
        for sub in ("mix", "s1", "s2"):
            os.makedirs(os.path.join(ldir, sub), exist_ok=True)
        for i in range(n_pairs):
            spk1, (uid1, p1, t1), spk2, (uid2, p2, t2) = pick_pair(
                condition, level, pool, genders, rng)
            seed = int(rng.integers(2 ** 31))
            mix, ref1, ref2 = generate_one(condition, level, p1, p2, seed)

            utt_id = f"{condition}_{level}_{i:02d}_{uid1}_{uid2}"
            paths = {sub: os.path.join(ldir, sub, utt_id + ".wav")
                     for sub in ("mix", "s1", "s2")}
            sf.write(paths["mix"], mix, SR, subtype="FLOAT")
            sf.write(paths["s1"], ref1, SR, subtype="FLOAT")
            sf.write(paths["s2"], ref2, SR, subtype="FLOAT")
            rows.append({
                "utt_id": utt_id, "condition": condition, "level": level,
                "mix": paths["mix"], "ref_1": paths["s1"], "ref_2": paths["s2"],
                "text_1": t1, "text_2": t2, "seed": seed,
                "src_1": p1, "src_2": p2,
            })
    with open(os.path.join(cdir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "condition", "level",
                                          "mix", "ref_1", "ref_2",
                                          "text_1", "text_2", "seed",
                                          "src_1", "src_2"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ----------------------------------------------------------------------------
def regen(manifest, utt_id, out_dir):
    """Rebuild one mixture purely from its manifest row (reproducibility)."""
    with open(manifest, newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["utt_id"] == utt_id)
    mix, _, _ = generate_one(row["condition"], row["level"],
                             row["src_1"], row["src_2"], row["seed"])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, utt_id + ".wav")
    sf.write(out, mix, SR, subtype="FLOAT")
    print(f"[regen] {utt_id} (seed={row['seed']}) -> {out}")
    print(f"[regen] original: {row['mix']}")
    return out, row["mix"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="testsets", help="Output root directory")
    p.add_argument("--n-pairs", type=int, default=3, help="Mixtures per level")
    p.add_argument("--librispeech", required=True,
                   help="Path to LibriSpeech test-clean directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--regen", default="", metavar="UTT_ID",
                   help="Regenerate one mixture from --regen-manifest")
    p.add_argument("--regen-manifest", default="")
    p.add_argument("--regen-out", default="regen_out")
    args = p.parse_args()

    if args.regen:
        regen(args.regen_manifest, args.regen, args.regen_out)
        return

    if "test-clean" not in os.path.abspath(args.librispeech):
        raise SystemExit("--librispeech must point at LibriSpeech test-clean")
    genders = load_speaker_genders(args.librispeech)
    pool = scan_pool(args.librispeech, genders)
    n_utts = sum(len(v) for v in pool.values())
    print(f"Pool: {len(pool)} speakers / {n_utts} utts "
          f"({MIN_UTT_S}-{MAX_UTT_S}s) from {args.librispeech}")

    rng = np.random.default_rng(args.seed)
    print(f"Building {args.n_pairs} mixture(s) per level into {args.out}/\n")
    print("Per-condition mixture counts:")
    print("-" * 40)
    total = 0
    for cond in args.conditions:
        if cond not in CONDITIONS:
            raise SystemExit(f"Unknown condition: {cond} (choose from {CONDITIONS})")
        c = build_condition(cond, args.out, args.n_pairs, pool, genders, rng)
        total += c
        print(f"  {cond:10s} {c:4d} mixtures  (levels: {', '.join(LEVELS[cond])})")
    print("-" * 40)
    print(f"  {'TOTAL':10s} {total:4d} mixtures")


if __name__ == "__main__":
    main()
