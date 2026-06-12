#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_v2.py - batched parallel separation-quality evaluation.

This is a non-breaking newer version of `evaluate.py`.

Adds:
1. Batched parallel evaluation over utterances via `--batch-size` and `--num-workers`.
2. Saving per-utterance results and final mean metrics to files.

The metric formulas and data-layout assumptions remain aligned with `evaluate.py`.
"""

import argparse
import csv
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torchaudio


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
    a = si_sdr(e0, r0) + si_sdr(e1, r1)
    b = si_sdr(e0, r1) + si_sdr(e1, r0)
    if a >= b:
        return (0, 1), [si_sdr(e0, r0).item(), si_sdr(e1, r1).item()]
    return (1, 0), [si_sdr(e0, r1).item(), si_sdr(e1, r0).item()]


def load_wav(path, target_sr):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() > 1:
        wav = wav.mean(-1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def match_len(*tensors):
    n = min(t.shape[-1] for t in tensors)
    return [t[..., :n] for t in tensors]


def read_transcripts(text_path):
    """Parse either a `text_spkN` or GT csv into key -> lowercase transcript."""
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


EST_SUBDIRS = ("spk1", "spk2")
REF_SUBDIRS = ("s1", "s2")


def discover_keys(output_dir):
    spk1_dir = os.path.join(output_dir, EST_SUBDIRS[0])
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(spk1_dir, "*.wav")))]


def estimate_paths(output_dir, key):
    return [os.path.join(output_dir, d, key) for d in EST_SUBDIRS]


def reference_paths(ref_dir, key):
    return [os.path.join(ref_dir, d, key) for d in REF_SUBDIRS]


def mixture_path(mix_dir, key):
    if not mix_dir:
        return None
    p = os.path.join(mix_dir, key)
    return p if os.path.exists(p) else None


def build_asr(model_name, device):
    from transformers import pipeline

    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    return pipeline("automatic-speech-recognition", model=model_name, device=dev)


def transcribe(asr, wav_8k, src_sr=8000):
    import jiwer  # noqa: F401

    wav16 = torchaudio.functional.resample(wav_8k, src_sr, 16000).numpy().astype("float32")
    return asr(wav16)["text"].strip().lower()


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def col_mean(rows, name):
    vals = [r[name] for r in rows if not np.isnan(r[name])]
    return float(np.mean(vals)) if vals else float("nan")


def build_summary(rows):
    return {
        "n": len(rows),
        "sisdr": col_mean(rows, "sisdr"),
        "sisdri": col_mean(rows, "sisdri"),
        "pesq": col_mean(rows, "pesq"),
        "stoi": col_mean(rows, "stoi"),
        "wer": col_mean(rows, "wer"),
    }


def save_results(rows, summary, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)

    rows_path = os.path.join(save_dir, f"{prefix}_utterance_metrics.csv")
    summary_json_path = os.path.join(save_dir, f"{prefix}_summary_mean.json")
    summary_txt_path = os.path.join(save_dir, f"{prefix}_summary_mean.txt")

    fieldnames = ["key", "perm", "sisdr", "sisdri", "pesq", "stoi", "wer"]
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(
            f"{'SUMMARY (n=' + str(summary['n']) + ')':<20}"
            f"{'SI-SDR':>10}{'SI-SDRi':>10}{'PESQ':>9}{'STOI':>9}{'WER':>9}\n"
        )
        f.write("-" * 78 + "\n")
        f.write(
            f"{'mean':<20}"
            f"{summary['sisdr']:>10.2f}{summary['sisdri']:>10.2f}"
            f"{summary['pesq']:>9.2f}{summary['stoi']:>9.3f}{summary['wer']:>9.2f}\n"
        )
        f.write("=" * 78 + "\n")

    return rows_path, summary_json_path, summary_txt_path


def evaluate_one_key(
    key,
    args,
    sr,
    do_wer,
    txt1,
    txt2,
    pesq_fn,
    stoi_fn,
    jiwer_module,
    asr=None,
    asr_lock=None,
):
    kk = os.path.splitext(key)[0]
    est_p = estimate_paths(args.output_dir, key)
    ref_p = reference_paths(args.ref_dir, key) if args.ref_dir else [None, None]
    if any(p is None or not os.path.exists(p) for p in ref_p):
        return {"skip": True, "key": key, "reason": "missing reference(s)"}

    e = [load_wav(p, sr) for p in est_p]
    r = [load_wav(p, sr) for p in ref_p]
    e[0], e[1], r[0], r[1] = match_len(e[0], e[1], r[0], r[1])

    perm, sisdr_vals = best_permutation_2spk(e, r)
    ref_aligned = [r[perm[0]], r[perm[1]]]

    sisdri_vals = []
    mix_p = mixture_path(args.mix_dir, key)
    if mix_p:
        m = load_wav(mix_p, sr)
        m, _, _, ra0, ra1 = match_len(m, e[0], e[1], ref_aligned[0], ref_aligned[1])
        base = [si_sdr(m, ra0).item(), si_sdr(m, ra1).item()]
        sisdri_vals = [sisdr_vals[0] - base[0], sisdr_vals[1] - base[1]]

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

    wer_v = []
    if do_wer and asr is not None:
        refs_txt = [txt1.get(kk), txt2.get(kk)]
        refs_txt_aligned = [refs_txt[perm[0]], refs_txt[perm[1]]]
        for est_i, ref_txt in zip(e, refs_txt_aligned):
            if not ref_txt:
                continue
            if asr_lock is None:
                hyp = transcribe(asr, est_i, sr)
            else:
                with asr_lock:
                    hyp = transcribe(asr, est_i, sr)
            wer_v.append(jiwer_module.wer(ref_txt, hyp) if (ref_txt or hyp) else 0.0)

    row = {
        "key": kk,
        "perm": f"{perm[0]}{perm[1]}",
        "sisdr": float(np.mean(sisdr_vals)),
        "sisdri": float(np.mean(sisdri_vals)) if sisdri_vals else float("nan"),
        "pesq": float(np.mean(pesq_v)) if pesq_v else float("nan"),
        "stoi": float(np.mean(stoi_v)) if stoi_v else float("nan"),
        "wer": float(np.mean(wer_v)) if wer_v else float("nan"),
    }
    return {"skip": False, "row": row}


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = args.sr

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
    asr_lock = threading.Lock() if do_wer else None
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

    batch_size = max(1, args.batch_size)
    num_workers = max(1, args.num_workers)

    rows = []
    print(
        f"Evaluating {len(keys)} utterance(s) from {args.output_dir} "
        f"with batch_size={batch_size}, num_workers={num_workers}\n"
    )

    for batch_idx, key_batch in enumerate(batched(keys, batch_size), start=1):
        print(f"[batch {batch_idx}] processing {len(key_batch)} utterance(s)")
        batch_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(
                    evaluate_one_key,
                    key,
                    args,
                    sr,
                    do_wer,
                    txt1,
                    txt2,
                    pesq_fn,
                    stoi_fn,
                    jiwer,
                    asr,
                    asr_lock,
                ): key
                for key in key_batch
            }
            for future in as_completed(futures):
                batch_results.append(future.result())

        for result in sorted(
            batch_results,
            key=lambda x: x["key"] if x.get("skip") else x["row"]["key"],
        ):
            if result.get("skip"):
                print(f"  [skip] {result['key']}: {result['reason']}")
                continue
            row = result["row"]
            rows.append(row)
            print(
                f"  {row['key'][:34]:34s} perm={row['perm']}  "
                f"SI-SDR={row['sisdr']:7.2f}  SI-SDRi={row['sisdri']:7.2f}  "
                f"PESQ={row['pesq']:5.2f}  STOI={row['stoi']:5.3f}  WER={row['wer']:5.2f}"
            )

    if not rows:
        raise SystemExit("No utterances were evaluated (no matching references).")

    summary = build_summary(rows)
    print("\n" + "=" * 78)
    print(f"{'SUMMARY (n=' + str(summary['n']) + ')':<20}"
          f"{'SI-SDR':>10}{'SI-SDRi':>10}{'PESQ':>9}{'STOI':>9}{'WER':>9}")
    print("-" * 78)
    print(f"{'mean':<20}"
          f"{summary['sisdr']:>10.2f}{summary['sisdri']:>10.2f}"
          f"{summary['pesq']:>9.2f}{summary['stoi']:>9.3f}{summary['wer']:>9.2f}")
    print("=" * 78)

    save_dir = args.save_dir or args.output_dir
    prefix = args.save_prefix or "evaluate_v2"
    rows_path, summary_json_path, summary_txt_path = save_results(rows, summary, save_dir, prefix)
    print(f"\nSaved utterance metrics to: {rows_path}")
    print(f"Saved mean summary json to: {summary_json_path}")
    print(f"Saved mean summary txt  to: {summary_txt_path}")

    return rows, summary


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
    p.add_argument("--batch-size", type=int, default=8, help="Number of utterances submitted per batch")
    p.add_argument("--num-workers", type=int, default=4, help="Parallel workers within each batch")
    p.add_argument("--save-dir", default="", help="Directory to save csv/json/txt results; defaults to output_dir")
    p.add_argument("--save-prefix", default="evaluate_v2", help="Prefix of saved result files")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
