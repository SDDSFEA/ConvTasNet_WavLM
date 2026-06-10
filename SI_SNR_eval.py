import argparse
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def parse_args():
    p = argparse.ArgumentParser(description="Compute SI-SNR on GPU and save results.")
    p.add_argument("--est_folder1", type=str, required=True, help="Estimated folder for speaker 1")
    p.add_argument("--est_folder2", type=str, required=True, help="Estimated folder for speaker 2")
    p.add_argument("--target_folder1", type=str, default="/lustre/users/shi/datasets/librimix/LibriMix/Libri2Mix/wav8k/max/test/s1", help="Target folder for speaker 1 (ground truth)")
    p.add_argument("--target_folder2", type=str, default="/lustre/users/shi/datasets/librimix/LibriMix/Libri2Mix/wav8k/max/test/s2", help="Target folder for speaker 2 (ground truth)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU evaluation")
    return p.parse_args()

def pit_si_snr_2spk(ests, refs, eps=1e-8):
    """
    ests: [B, 2, T]
    refs: [B, 2, T]
    return: [B]
    """
    # zero-mean
    ests = ests - ests.mean(dim=2, keepdim=True)
    refs = refs - refs.mean(dim=2, keepdim=True)

    def si_snr_pair(x, s):
        # x, s: [B, T]
        dot = torch.sum(x * s, dim=1, keepdim=True)
        s_energy = torch.sum(s ** 2, dim=1, keepdim=True) + eps
        s_target = dot / s_energy * s
        e_noise = x - s_target
        return 10 * torch.log10(
            torch.sum(s_target ** 2, dim=1) /
            (torch.sum(e_noise ** 2, dim=1) + eps)
        )

    # 两种排列
    snr_00 = si_snr_pair(ests[:, 0], refs[:, 0]) \
           + si_snr_pair(ests[:, 1], refs[:, 1])

    snr_01 = si_snr_pair(ests[:, 0], refs[:, 1]) \
           + si_snr_pair(ests[:, 1], refs[:, 0])

    return torch.max(snr_00, snr_01) / 2

import os
import glob
import torch
import torchaudio
import numpy as np
from tqdm import tqdm


def calculate_si_snr_gpu(
    est_folder1,
    est_folder2,
    target_folder1,
    target_folder2,
    sample_rate=8000,
    batch_size=16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    est1_files = sorted(glob.glob(os.path.join(est_folder1, "*.wav")))
    est2_files = sorted(glob.glob(os.path.join(est_folder2, "*.wav")))
    tgt1_files = sorted(glob.glob(os.path.join(target_folder1, "*.wav")))
    tgt2_files = sorted(glob.glob(os.path.join(target_folder2, "*.wav")))

    print(len(est1_files) , len(est2_files) , len(tgt1_files) , len(tgt2_files))
    assert len(est1_files) == len(est2_files) == len(tgt1_files) == len(tgt2_files)
    total = len(est1_files)

    results = []

    for start in tqdm(range(0, total, batch_size), desc="SI-SNR (GPU)"):
        end = min(start + batch_size, total)

        ests, refs = [], []

        # -------- load --------
        for i in range(start, end):
            e1, sr = torchaudio.load(est1_files[i])
            e2, _  = torchaudio.load(est2_files[i])
            t1, _  = torchaudio.load(tgt1_files[i])
            t2, _  = torchaudio.load(tgt2_files[i])

            assert sr == sample_rate

            e1 = e1.mean(0)
            e2 = e2.mean(0)
            t1 = t1.mean(0)
            t2 = t2.mean(0)

            ests.append((e1, e2))
            refs.append((t1, t2))

        # -------- padding --------
        max_len = max(
            max(e1.size(0), e2.size(0), t1.size(0), t2.size(0))
            for (e1, e2), (t1, t2) in zip(ests, refs)
        )

        def pad(x, length):
            return torch.nn.functional.pad(x, (0, length - x.size(0)))

        ests = torch.stack([
            torch.stack([pad(e1, max_len), pad(e2, max_len)])
            for (e1, e2) in ests
        ]).to(device)   # [B, 2, T]

        refs = torch.stack([
            torch.stack([pad(t1, max_len), pad(t2, max_len)])
            for (t1, t2) in refs
        ]).to(device)

        # -------- compute --------
        with torch.no_grad():
            sisnr = pit_si_snr_2spk(ests, refs)

        results.extend(sisnr.cpu().tolist())

    return np.array(results)


if __name__ == "__main__":
    args = parse_args()

    si_snr = calculate_si_snr_gpu(
        args.est_folder1,
        args.est_folder2,
        args.target_folder1,
        args.target_folder2,
        batch_size=64
    )

    print(f"Avg SI-SNR: {si_snr.mean():.2f} dB")
    np.save("si_snr_gpu.npy", si_snr)
