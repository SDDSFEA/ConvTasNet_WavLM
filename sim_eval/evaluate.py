#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — separation-quality evaluation for this ConvTasNet+WavLM repo.

Computes SI-SDR, SI-SDRi, PESQ, STOI and WER over a directory of separated
audio, with a SI-SDR-based permutation (PIT) to align estimates to references.

================================================================================
REAL MODEL OUTPUT LAYOUT  (verified from Separation_nets.py :: Separation.inference)
================================================================================
The model's inference (`Separation_nets.py`, used by `inferece.sh`) writes, for a
save path `P`:

    P/
      spk1/<key>.wav      # estimated source 1
      spk2/<key>.wav      # estimated source 2

  * NUMBER OF SOURCES : 2 fixed folders, literally named "spk1" and "spk2"
                        (`os.makedirs(file_path+'/spk'+str(index))`, index = 1,2).
  * FILENAME (<key>)  : identical to the *mixture* utterance key from the .scp,
                        e.g. `1272-128104-0000_2035-147961-0014.wav`. The SAME
                        basename is reused across spk1/, spk2/, and the ground
                        truth s1/, s2/ folders — that is the join key.
  * SAMPLE RATE       : 8000 Hz  (`write_wav(filename, s, 8000)`).
  * CHANNELS / DTYPE  : mono, float32 wav (torchaudio.save), peak-renormalised so
                        the estimate's inf-norm matches the input mixture's.

Ground-truth references (LibriMix Libri2Mix, "max" mode) live separately:

    .../LibriMix/Libri2Mix/wav8k/max/<set>/s1/<key>.wav      # ref source 1
    .../LibriMix/Libri2Mix/wav8k/max/<set>/s2/<key>.wav      # ref source 2
    .../LibriMix/Libri2Mix/wav8k/max/<set>/mix_both/<key>.wav   # noisy mixture
    .../LibriMix/Libri2Mix/wav8k/max/<set>/mix_clean/<key>.wav  # clean mixture

Reference transcripts (for WER) live in this repo under data/text/<set>/, in two
interchangeable formats keyed by the same <key>:
    text_spk1 / text_spk2 : "<key-without-ext> UPPERCASE TRANSCRIPT"
    GT_<set>_spk1.csv      : "ID,Speaker" rows with ID="<key>.wav", lowercase text
================================================================================
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch
import torchaudio


# ----------------------------------------------------------------------------
# Metric math + SI-SDR permutation logic  (do NOT alter the formulas here)
# ----------------------------------------------------------------------------
def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR (dB) between 1-D tensors est and ref."""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    s_target = alpha * ref
    e_noise = est - s_target
    return 10.0 * torch.log10(
        (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    )


def best_permutation_2spk(ests, refs):
    """
    SI-SDR-based PIT for 2 sources.
    ests / refs : list of two 1-D tensors (already length-matched).
    Returns (perm, per_source_sisdr) where perm is the index order of refs that
    maximises summed SI-SDR, and per_source_sisdr are the SI-SDRs under that perm.
    """
    e0, e1 = ests
    r0, r1 = refs
    # permutation (0,1): est0->ref0, est1->ref1
    a = si_sdr(e0, r0) + si_sdr(e1, r1)
    # permutation (1,0): est0->ref1, est1->ref0
    b = si_sdr(e0, r1) + si_sdr(e1, r0)
    if a >= b:
        return (0, 1), [si_sdr(e0, r0).item(), si_sdr(e1, r1).item()]
    return (1, 0), [si_sdr(e0, r1).item(), si_sdr(e1, r0).item()]


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def load_wav(path, target_sr):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() > 1:
        wav = wav.mean(-1)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def match_len(*tensors):
    n = min(t.shape[-1] for t in tensors)
    return [t[..., :n] for t in tensors]


def read_transcripts(text_path):
    """Parse either a `text_spkN` (key TEXT) or a GT csv (ID,Speaker) file.
    Returns dict keyed by basename WITHOUT extension -> lowercase transcript."""
    if not text_path or not os.path.exists(text_path):
        return {}
    out = {}
    if text_path.endswith(".csv"):
        with open(text_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = os.path.splitext(row["ID"])[0]
                out[key] = row["Speaker"].strip().lower()
    else:
        with open(text_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                key = os.path.splitext(parts[0])[0]
                out[key] = parts[1].strip().lower()
    return out


# ----------------------------------------------------------------------------
# ============================  ADAPT-THIS  ==================================
# How estimates are discovered in a model OUTPUT directory, and how each
# estimate key is mapped back to its references / mixture / transcript.
#
# For THIS repo the inference layout is `<out>/spk1/<key>.wav`,
# `<out>/spk2/<key>.wav` (see header). If you change Separation_nets.py's
# output convention, edit ONLY the four hooks below.
# ----------------------------------------------------------------------------
EST_SUBDIRS = ("spk1", "spk2")  # ADAPT-THIS: estimate folders, in source order
REF_SUBDIRS = ("s1", "s2")      # ADAPT-THIS: ground-truth source folders


def discover_keys(output_dir):
    """ADAPT-THIS: list utterance keys present in the model output dir."""
    spk1_dir = os.path.join(output_dir, EST_SUBDIRS[0])
    keys = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(spk1_dir, "*.wav")))]
    return keys


def estimate_paths(output_dir, key):
    """ADAPT-THIS: absolute paths of the two estimates for `key`."""
    return [os.path.join(output_dir, d, key) for d in EST_SUBDIRS]


def reference_paths(ref_dir, key):
    """ADAPT-THIS: absolute paths of the two ground-truth sources for `key`."""
    return [os.path.join(ref_dir, d, key) for d in REF_SUBDIRS]


def mixture_path(mix_dir, key):
    """ADAPT-THIS: absolute path of the mixture for `key` (None if unavailable)."""
    if not mix_dir:
        return None
    p = os.path.join(mix_dir, key)
    return p if os.path.exists(p) else None
# ==========================  END ADAPT-THIS  ================================


# ----------------------------------------------------------------------------
# Optional ASR for WER
# ----------------------------------------------------------------------------
def build_asr(model_name, device):
    from transformers import pipeline
    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    return pipeline("automatic-speech-recognition", model=model_name, device=dev)


def transcribe(asr, wav_8k, src_sr=8000):
    import jiwer  # noqa: F401  (ensures dependency present early)
    wav16 = torchaudio.functional.resample(wav_8k, src_sr, 16000).numpy().astype("float32")
    text = asr(wav16)["text"].strip().lower()
    return text


# ----------------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------------
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = args.sr

    # Resolve a co-located `_refs` fixture if the user did not pass refs and one
    # exists next to the output dir (used by `run_inference.sh fixture`).
    auto_refs = os.path.join(args.output_dir, "_refs")
    if not args.ref_dir and os.path.isdir(auto_refs):
        args.ref_dir = auto_refs
        if not args.mix_dir and os.path.isdir(os.path.join(auto_refs, "mix")):
            args.mix_dir = os.path.join(auto_refs, "mix")
        if not args.text_spk1 and os.path.exists(os.path.join(auto_refs, "text_spk1")):
            args.text_spk1 = os.path.join(auto_refs, "text_spk1")
        if not args.text_spk2 and os.path.exists(os.path.join(auto_refs, "text_spk2")):
            args.text_spk2 = os.path.join(auto_refs, "text_spk2")

    keys = discover_keys(args.output_dir)
    if args.limit:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit(f"No estimates found under {args.output_dir}/{EST_SUBDIRS[0]}/*.wav")

    do_wer = not args.no_wer and bool(args.text_spk1 or args.text_spk2)
    asr = build_asr(args.asr_model, device) if do_wer else None
    txt1 = read_transcripts(args.text_spk1)
    txt2 = read_transcripts(args.text_spk2)

    try:
        from pesq import pesq as pesq_fn
    except Exception:
        pesq_fn = None
    try:
        from pystoi import stoi as stoi_fn
    except Exception:
        stoi_fn = None
    import jiwer

    rows = []
    print(f"Evaluating {len(keys)} utterance(s) from {args.output_dir}\n")
    for key in keys:
        kk = os.path.splitext(key)[0]
        est_p = estimate_paths(args.output_dir, key)
        ref_p = reference_paths(args.ref_dir, key) if args.ref_dir else [None, None]
        if any(p is None or not os.path.exists(p) for p in ref_p):
            print(f"  [skip] {key}: missing reference(s)")
            continue

        e = [load_wav(p, sr) for p in est_p]
        r = [load_wav(p, sr) for p in ref_p]
        e[0], e[1], r[0], r[1] = match_len(e[0], e[1], r[0], r[1])

        perm, sisdr_vals = best_permutation_2spk(e, r)
        # references aligned to estimates under the chosen permutation
        ref_aligned = [r[perm[0]], r[perm[1]]]

        # SI-SDRi vs the mixture-as-trivial-estimate baseline
        sisdri_vals = []
        mix_p = mixture_path(args.mix_dir, key)
        if mix_p:
            m = load_wav(mix_p, sr)
            m, e0a, e1a, ra0, ra1 = match_len(m, e[0], e[1], ref_aligned[0], ref_aligned[1])
            base = [si_sdr(m, ra0).item(), si_sdr(m, ra1).item()]
            sisdri_vals = [sisdr_vals[0] - base[0], sisdr_vals[1] - base[1]]

        # PESQ / STOI per matched source, averaged over the 2 sources
        pesq_v, stoi_v = [], []
        for est_i, ref_i in zip(e, ref_aligned):
            ee, rr = match_len(est_i, ref_i)
            ee_np, rr_np = ee.numpy().astype("float64"), rr.numpy().astype("float64")
            if pesq_fn is not None:
                try:
                    mode = "nb" if sr == 8000 else "wb"
                    pesq_v.append(pesq_fn(sr, rr_np, ee_np, mode))
                except Exception:
                    pass
            if stoi_fn is not None:
                try:
                    stoi_v.append(stoi_fn(rr_np, ee_np, sr, extended=False))
                except Exception:
                    pass

        # WER per matched source via ASR (estimate) vs reference transcript
        wer_v = []
        if do_wer:
            refs_txt = [txt1.get(kk), txt2.get(kk)]
            refs_txt_aligned = [refs_txt[perm[0]], refs_txt[perm[1]]]
            for est_i, ref_txt in zip(e, refs_txt_aligned):
                if not ref_txt:
                    continue
                hyp = transcribe(asr, est_i, sr)
                wer_v.append(jiwer.wer(ref_txt, hyp) if (ref_txt or hyp) else 0.0)

        row = {
            "key": kk,
            "perm": f"{perm[0]}{perm[1]}",
            "sisdr": float(np.mean(sisdr_vals)),
            "sisdri": float(np.mean(sisdri_vals)) if sisdri_vals else float("nan"),
            "pesq": float(np.mean(pesq_v)) if pesq_v else float("nan"),
            "stoi": float(np.mean(stoi_v)) if stoi_v else float("nan"),
            "wer": float(np.mean(wer_v)) if wer_v else float("nan"),
        }
        rows.append(row)
        print(
            f"  {row['key'][:34]:34s} perm={row['perm']}  "
            f"SI-SDR={row['sisdr']:7.2f}  SI-SDRi={row['sisdri']:7.2f}  "
            f"PESQ={row['pesq']:5.2f}  STOI={row['stoi']:5.3f}  WER={row['wer']:5.2f}"
        )

    if not rows:
        raise SystemExit("No utterances were evaluated (no matching references).")

    def col_mean(name):
        vals = [r[name] for r in rows if not np.isnan(r[name])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 78)
    print(f"{'SUMMARY (n=' + str(len(rows)) + ')':<20}"
          f"{'SI-SDR':>10}{'SI-SDRi':>10}{'PESQ':>9}{'STOI':>9}{'WER':>9}")
    print("-" * 78)
    print(f"{'mean':<20}"
          f"{col_mean('sisdr'):>10.2f}{col_mean('sisdri'):>10.2f}"
          f"{col_mean('pesq'):>9.2f}{col_mean('stoi'):>9.3f}{col_mean('wer'):>9.2f}")
    print("=" * 78)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="Model output dir containing spk1/ and spk2/")
    p.add_argument("--ref-dir", default="", help="Dir containing s1/ and s2/ ground-truth sources")
    p.add_argument("--mix-dir", default="", help="Dir of mixture wavs (for SI-SDRi); optional")
    p.add_argument("--text-spk1", default="", help="Reference transcript file/csv for source 1")
    p.add_argument("--text-spk2", default="", help="Reference transcript file/csv for source 2")
    p.add_argument("--sr", type=int, default=8000, help="Sample rate (model writes 8000 Hz)")
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N utterances")
    p.add_argument("--asr-model", default="openai/whisper-tiny.en", help="HF ASR model for WER")
    p.add_argument("--no-wer", action="store_true", help="Skip ASR/WER")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
