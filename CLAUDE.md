# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **2-speaker speech separation** built on Conv-TasNet, enhanced with **WavLM** semantic features injected via cross-attention. Trained/evaluated on LibriMix (Libri2Mix), 8 kHz, `max` mode, `clean` (and some `noise`) subsets. This is an experiment-driven repo: many model files are alternative fusion strategies, not a single canonical model.

## Commands

Training (uses the venv python; activate or call it directly):
```bash
# Single hardcoded model — edit the `from ... import ConvTasNet` line in train.py to swap architecture
python train.py --opt options/train/train_clean100_WavLM_film.yml

# Multi-model dispatcher — picks architecture by name (see build_model / if-chain)
python train_nets.py      --opt options/train/train_clean100_WavLM_dwconvFuse_film.yml --model_net Net_dwAtt_film
python train_nets_Attn.py --opt <yml> --model_net <NetName>   # same dispatch, uses trainer_attn (selective freezing)
```
`--model_net` values: `Net_dwAtt1`, `Net_dwAtt2`, `Net_dwAtt_film`, `Net_film`, `Net_gate`, `Net_up`, `Net_dwAtt1_wogate`, `Net_dwAtt1_nogate`, `Net_dwAtt1_woshare`.

Inference + evaluation (batch over an `.scp` list, then score):
```bash
# End-to-end on a SLURM cluster (edit net_name / model_path at top)
bash inferece.sh

# Or directly:
python Separation_nets.py -mix_scp tt_mix.scp -yaml <yml> -model best.pt -gpuid 0 -model_net Net_dwAtt_film -save_path enhanced/<dir>
python SI_SNR_eval.py --est_folder1 .../spk1 --est_folder2 .../spk2 --target_folder1 .../s1 --target_folder2 .../s2
```

WER (ASR-based) scoring: `whisper_ASR.py` / `compute_wer_whisper.py` (Whisper transcription) → `compute_wer.py` (Kaldi-style WER scorer; run as a script with hyp/ref args).

Data prep: `create_scp.py` walks LibriMix wav directories and writes `key <path>` `.scp` files (edit the hardcoded paths inside). SLURM submission via `submit.sh` / `sbatch run_*.sh`.

There is **no test suite, linter, or build step**. Several model files have an `if __name__ == "__main__": test()` block that runs a random-tensor forward pass for a quick sanity check (`python Conv_TasNet_wavlm_dwconvFuse.py`).

## Architecture

**Forward pass** (e.g. `Conv_TasNet_wavlm_dwconvFuse.py`, the reference variant):
1. **Audio path** — 1-D conv `encoder` → cLN → `bottleneck` (N=512 → B=128).
2. **Semantic path** — `WavLMencoder` (wraps `modeling_wavlm.WavLMModel.from_pretrained`) produces `[B, T_sem, D]`. WavLM is **frozen by default** (`freeze_wavlm=True`).
3. **SeparationModule** — `R` repeats × `X` `Conv1D_Block`s (TCN with exponentially increasing dilation). Semantic features are injected inside each block via `CrossAttnDelta` (Q from audio, K/V from WavLM) and added back through a `GatedResidual` (sigmoid gate, init ≈ -2 so injection starts near-off). Attention is **shared per repeat** by default.
4. Mask head → apply masks to encoder output → `ConvTrans1D` decoder → list of `num_spks` waveforms.

**Loss**: PIT (permutation-invariant) SI-SNR — `SI_SNR.si_snr_loss`. `SI_SNR_eval.py` has its own standalone 2-spk PIT SI-SNR for offline scoring.

**Model variants** (`Conv_TasNet_wavlm_*.py`) differ only in the fusion mechanism. Naming decodes the strategy: `dwconvFuse` (inject after depthwise conv), `film` (FiLM conditioning), `gate`/`wogate`/`nogate` (gated vs ungated residual), `woshare` (per-block instead of per-repeat attention), `up`/`linearAlign` (semantic upsampling/alignment to audio rate), `scconvFuse`/`att2`. `Conv_TasNet.py` is the plain Conv-TasNet baseline (no WavLM). When changing architecture, you generally **create or pick a model file and point the training entry's import / `--model_net` at it** — the models are not selected by config.

**The DataLoader is non-standard** (`DataLoaders.py`): each utterance is split into 4 s (32000-sample) chunks at load time. The inner `DataLoader` uses `batch_size // 2` and a custom collate that re-chunks, so the effective batch is reassembled in `__iter__`. Consequence: **`batch_size` should be even**, and one "sample" expands into a variable number of training chunks. `DataLoaders_max.py` is a no-chunking variant.

## Config

YAML files in `options/train/*.yml`, parsed by `options/option.py` (`parse`). Sections: `net_conf` (passed as `**kwargs` to the model `__init__` — must match that model's signature), `datasets` (scp paths + `batch_size`/`num_workers`), `train` (lr schedule via `ReduceLROnPlateau`, `clip_norm`, `checkpoint` dir name), `optimizer_kwargs`, `resume` (`path` + `resume_state` flag), `gpu_ids`. Filenames encode the experiment: `train_clean100_WavLM_<fusion>[_<ablation>].yml`.

## Trainers

`trainer.py` is the standard one (full fine-tune of unfrozen params, data-parallel, SI-SNR, plateau LR, save `best.pt`/`last.pt`). Variants change **which parameters train**:
- `trainer_attn.py` — `mark_only_added_modules_trainable`: freezes WavLM **and** the original Conv-TasNet backbone, trains only the newly added fusion/attention modules (identified by parameter-name prefixes/keywords).
- `trainer_adap.py` — parameter-group / selective `requires_grad` freezing (`get_audio_parameters`, `get_attention_parameters`, `get_wavlmencoder_parameters` helpers on the model).
- `trainer_baseline.py` — for the no-WavLM baseline.

Models expose `get_audio_parameters()` / `get_attention_parameters()` / `get_wavlmencoder_parameters()` so trainers can target subsets.

## Important quirks

- **Hardcoded absolute paths** are everywhere: `/lustre/users/shi/...` dataset/checkpoint roots in `.yml` files, `inferece.sh`, `create_scp.py`; and **fallback resume checkpoint paths** baked into `trainer.py`/`trainer_attn.py` (`/home/student/zt/...`). Expect to edit these for any new environment.
- Some entry/inference files import `from ConvTasNet_separation.Conv_TasNet_wavlm_dwconvFuse_wogate import ...` — that package does **not** exist in this repo; those specific `--model_net` branches (`Net_dwAtt1_nogate`) will fail to import unless the path is fixed.
- `option.parse` takes a misspelled kwarg `is_tain` (not `is_train`) — keep it when calling.
- `train*.py` do `sys.path.append('./options')` to import `option`; run training from the repo root.
- `.pt` checkpoints, `*.npy`, `result*/`, and `wavlm_large_pretrained/` are git-ignored.
