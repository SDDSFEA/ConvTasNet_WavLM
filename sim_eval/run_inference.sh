#!/bin/bash
# ===========================================================================
# run_inference.sh — driver for separation inference, synthesis and evaluation
# for the ConvTasNet + WavLM repo.
#
# Subcommands:
#   run      <mix_scp> <save_path>            run the real model (Separation_nets.py)
#   eval     <output_dir> [evaluate.py args]  score a model-output dir (evaluate.py)
#   run_eval <mix_scp> <save_path> [eval args] run the model THEN score it in one shot
#   synth    [n_pairs] [out_dir]              build the 4-condition eval sets
#   fixture  <dir> [n]                        synthetic model-format output for testing eval
#
# `run_eval` defaults its references to the LibriMix paths below (override per
# call with --ref-dir/--mix-dir/--text-spk1/--text-spk2, forwarded to evaluate.py).
# ===========================================================================
set -euo pipefail

PY="${PYTHON:-python}"
# This script lives in <repo>/sim_eval/. HERE = that sub-dir; REPO_ROOT = its parent
# (where Separation_nets.py, options/, data/ live). synth/eval/fixture run from HERE;
# the real model (run_model) runs from REPO_ROOT.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"

# ---------------------------------------------------------------------------
# ===============================  ADAPT-THIS  ==============================
# run_model(): invoke the repo's REAL separation inference. The output layout
# this produces (<save_path>/spk1/<key>.wav, <save_path>/spk2/<key>.wav, 8 kHz,
# 2 sources) is exactly what `eval` consumes — see evaluate.py's header.
#
# Set these to your trained checkpoint / config / architecture before use.
# ---------------------------------------------------------------------------
MODEL_NET="${MODEL_NET:-Net_dwAtt_film}"                                              # ADAPT-THIS: --model_net
YAML="${YAML:-$REPO_ROOT/options/train/train_clean100_WavLM_dwconvFuse_film.yml}"     # ADAPT-THIS: -yaml
MODEL_CKPT="${MODEL_CKPT:-/lustre/users/shi/datasets/librimix/ConvTasNet/best.pt}"    # ADAPT-THIS: -model
GPUID="${GPUID:-0}"                                                                   # ADAPT-THIS: -gpuid

# References used by `run_eval` when not overridden on the command line (these
# mirror SI_SNR_eval.py / inferece.sh). <set> = test|dev — set EVAL_SET to switch.
EVAL_SET="${EVAL_SET:-test}"
LIBRIMIX_ROOT="${LIBRIMIX_ROOT:-/lustre/users/shi/datasets/librimix/LibriMix/Libri2Mix/wav8k/max}"  # ADAPT-THIS
REF_DIR="${REF_DIR:-$LIBRIMIX_ROOT/$EVAL_SET}"                      # contains s1/ and s2/
MIX_DIR="${MIX_DIR:-$LIBRIMIX_ROOT/$EVAL_SET/mix_both}"             # for SI-SDRi
TEXT_SPK1="${TEXT_SPK1:-$REPO_ROOT/data/text/$EVAL_SET/text_spk1}"  # for WER (if present)
TEXT_SPK2="${TEXT_SPK2:-$REPO_ROOT/data/text/$EVAL_SET/text_spk2}"

run_model() {
  local mix_scp="$1" save_path="$2"
  echo "[run_model] net=$MODEL_NET ckpt=$MODEL_CKPT -> $save_path"
  # Separation_nets.py imports repo-root modules and does sys.path.append('./options'),
  # so it must run from REPO_ROOT. save_path is made absolute so output lands where asked.
  case "$save_path" in /*) : ;; *) save_path="$HERE/$save_path" ;; esac
  ( cd "$REPO_ROOT" && "$PY" Separation_nets.py \
      -mix_scp "$mix_scp" \
      -yaml "$YAML" \
      -model "$MODEL_CKPT" \
      -gpuid "$GPUID" \
      -model_net "$MODEL_NET" \
      -save_path "$save_path" )
  # Produces: $save_path/spk1/<key>.wav and $save_path/spk2/<key>.wav (8 kHz).
}
# ============================  END ADAPT-THIS  =============================

cmd_run() {
  [ $# -ge 2 ] || { echo "usage: $0 run <mix_scp> <save_path>" >&2; exit 2; }
  run_model "$1" "$2"
}

# Inference + evaluation in a single shot.
cmd_run_eval() {
  [ $# -ge 2 ] || { echo "usage: $0 run_eval <mix_scp> <save_path> [evaluate.py args]" >&2; exit 2; }
  local mix_scp="$1" save_path="$2"; shift 2
  run_model "$mix_scp" "$save_path"
  echo "[run_eval] scoring $save_path against set=$EVAL_SET"
  local ref_args=()
  [ -d "$REF_DIR" ]    && ref_args+=(--ref-dir "$REF_DIR")
  [ -d "$MIX_DIR" ]    && ref_args+=(--mix-dir "$MIX_DIR")
  [ -f "$TEXT_SPK1" ]  && ref_args+=(--text-spk1 "$TEXT_SPK1")
  [ -f "$TEXT_SPK2" ]  && ref_args+=(--text-spk2 "$TEXT_SPK2")
  "$PY" evaluate.py "$save_path" "${ref_args[@]}" "$@"
}

cmd_synth() {
  local n="${1:-3}" out="${2:-synth_data}"
  "$PY" synthesize_datasets.py --n-pairs "$n" --out "$out"
}

cmd_eval() {
  [ $# -ge 1 ] || { echo "usage: $0 eval <output_dir> [extra evaluate.py args]" >&2; exit 2; }
  local out_dir="$1"; shift
  "$PY" evaluate.py "$out_dir" "$@"
}

# Build a synthetic model-format output dir (+ co-located _refs) so `eval` can
# be exercised end-to-end without a GPU/checkpoint/LibriMix data.
cmd_fixture() {
  local dir="${1:-fixture_out}" n="${2:-5}"
  "$PY" - "$dir" "$n" <<'PYEOF'
import os, sys
import numpy as np, soundfile as sf
from synthesize_datasets import synth_voice, SR

dir_, n = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(123)
for sub in ("spk1", "spk2", "_refs/s1", "_refs/s2", "_refs/mix"):
    os.makedirs(os.path.join(dir_, *sub.split("/")), exist_ok=True)

t1, t2 = [], []
for i in range(n):
    a1 = synth_voice(rng, 3.0)
    a2 = synth_voice(rng, 3.0)
    m = min(len(a1), len(a2)); a1, a2 = a1[:m], a2[:m]
    mix = (a1 + a2).astype("float32")
    key = f"utt_{i:04d}.wav"; kk = key[:-4]
    sf.write(os.path.join(dir_, "_refs", "s1", key), a1, SR)
    sf.write(os.path.join(dir_, "_refs", "s2", key), a2, SR)
    sf.write(os.path.join(dir_, "_refs", "mix", key), mix, SR)
    # estimates: reference + ~10 dB noise; swap order on odd indices to exercise PIT
    def corrupt(x):
        nse = rng.standard_normal(len(x)).astype("float32")
        sc = np.sqrt(np.mean(x**2) / (np.mean(nse**2) + 1e-12) / 10.0)
        return (x + sc * nse).astype("float32")
    e1, e2 = corrupt(a1), corrupt(a2)
    if i % 2 == 1:
        e1, e2 = e2, e1  # estimates in swapped order -> PIT must recover it
    sf.write(os.path.join(dir_, "spk1", key), e1, SR)
    sf.write(os.path.join(dir_, "spk2", key), e2, SR)
    t1.append(f"{kk} the quick brown fox jumps over the lazy dog")
    t2.append(f"{kk} she sells sea shells by the sea shore")

open(os.path.join(dir_, "_refs", "text_spk1"), "w").write("\n".join(t1) + "\n")
open(os.path.join(dir_, "_refs", "text_spk2"), "w").write("\n".join(t2) + "\n")
print(f"[fixture] wrote {n} model-format utterances + _refs into {dir_}/")
PYEOF
}

main() {
  [ $# -ge 1 ] || { echo "usage: $0 {run|eval|run_eval|synth|fixture} ..." >&2; exit 2; }
  local sub="$1"; shift
  case "$sub" in
    run)      cmd_run "$@" ;;
    eval)     cmd_eval "$@" ;;
    run_eval) cmd_run_eval "$@" ;;
    synth)    cmd_synth "$@" ;;
    fixture)  cmd_fixture "$@" ;;
    *) echo "unknown subcommand: $sub" >&2; exit 2 ;;
  esac
}

main "$@"
