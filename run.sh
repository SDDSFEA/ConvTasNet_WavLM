#!/bin/bash
#SBATCH --job-name=1b-2spk_n
#SBATCH --partition=002-partition-all
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/users/shi/audio_llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

source /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/activate


# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt ./options/train/train_clean100_WavLM_dwconvFuse_att2.yml --model_net Net_dwAtt1
# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt ./options/train/train_clean100_WavLM_dwconvFuse_att2.yml --model_net Net_dwAtt2
# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt ./options/train/train_clean100_WavLM_dwconvFuse_film.yml --model_net Net_dwAtt_film
# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt options/train/train_clean100_WavLM_film.yml --model_net Net_film
# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt options/train/train_clean100_WavLM_gate.yml --model_net Net_gate
# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets.py --opt options/train/train_clean100_WavLM_up.yml --model_net Net_up

# /lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets_unfreeze.py --opt ./options/train/train_clean100_WavLM_dwconvFuse_att2_unfreeze.yml --model_net Net_dwAtt2
/lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python train_nets_unfreeze.py --opt options/train/train_clean100_WavLM_up_unfreeze.yml --model_net Net_up

