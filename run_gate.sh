#!/bin/bash
#SBATCH --job-name=1b-2spk_n
#SBATCH --partition=002-partition-all
#SBATCH --gpus=2
#SBATCH --container-image=/lustre/users/shi/audio_llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

set -euo pipefail

# (可选) 打印环境，确认在容器内
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
which python || true
nvidia-smi -L || true

source /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/activate
cd /lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt
/lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt options/train/train_clean100_WavLM_gate.yml --model_net Net_gate



