#!/bin/bash
set -euo pipefail

sub_set=test
out_dir=Net_dwAtt_film_clean100
net_name="Net_dwAtt_film"
model_path=/lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/Conv-TasNet-clean100-WavLM-dwconvFuse-gate-film
# model_path=/lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/Conv-TasNet-clean100-WavLM-dwconvFuse-gate
if [[ "$sub_set" == "dev" ]]; then
  file_name="cv_mix.scp"
elif [[ "$sub_set" == "test" ]]; then
  file_name="tt_mix.scp"
else
  echo "ERROR: sub_set must be dev or test, got: $sub_set" >&2
  exit 1
fi

:<<COMMENT
/lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python Separation.py \
	-mix_scp /lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/data/audio_scp_8k/test/clean/tt_mix.scp \
	-yaml ./options/train/train_clean360.yml \
	-model /lustre/users/shi/datasets/librimix/ckpt_convtasnet/best.pt \
	-gpuid 0 \
	-save_path enhanced/baseline_clean100
COMMENT


:<<COMMENT
Net_dwAtt1 ./options/train/train_clean100_WavLM_dwconvFuse_att2.yml
Net_dwAtt2 ./options/train/train_clean100_WavLM_dwconvFuse_att2.yml
Net_dwAtt_film ./options/train/train_clean100_WavLM_dwconvFuse_film.yml
Net_film options/train/train_clean100_WavLM_film.yml
Net_gate options/train/train_clean100_WavLM_gate.yml
Net_up options/train/train_clean100_WavLM_up.yml 
COMMENT

case "$net_name" in
  Net_dwAtt1)     opt_yaml="./options/train/train_clean100_WavLM_dwconvFuse_att2.yml" ;;
  Net_dwAtt2)     opt_yaml="./options/train/train_clean100_WavLM_dwconvFuse_att2.yml" ;;
  Net_dwAtt_film) opt_yaml="./options/train/train_clean100_WavLM_dwconvFuse_film.yml" ;;
  Net_film)       opt_yaml="options/train/train_clean100_WavLM_film.yml" ;;
  Net_gate)       opt_yaml="options/train/train_clean100_WavLM_gate.yml" ;;
  Net_up)         opt_yaml="options/train/train_clean100_WavLM_up.yml" ;;
  *)
    echo "ERROR: Unknown net_name: $net_name" >&2
    exit 1
    ;;
esac
echo "net_name=$net_name"
echo "opt_yaml=$opt_yaml"
echo "out_dir=$out_dir"
echo "model_path=$model_path"

/lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python Separation_nets.py \
        -mix_scp /lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/data/audio_scp_8k/${sub_set}/clean/${file_name} \
        -yaml $opt_yaml \
        -model ${model_path}/best.pt \
        -gpuid 0 \
	-model_net $net_name \
        -save_path enhanced/${out_dir}/${sub_set}

/lustre/users/shi/toolkits/m_speaker_llm/venv/bin/python SI_SNR_eval.py \
	--est_folder1 /lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/enhanced/${out_dir}/${sub_set}/spk1 \
	--est_folder2 /lustre/users/shi/datasets/librimix/ConvTasNet/ConvTasNet_Separation_WavLM_CAtt/enhanced/${out_dir}/${sub_set}/spk2 \
	--target_folder1 /lustre/users/shi/datasets/librimix/LibriMix/Libri2Mix/wav8k/max/${sub_set}/s1 \
	--target_folder2 /lustre/users/shi/datasets/librimix/LibriMix/Libri2Mix/wav8k/max/${sub_set}/s2

