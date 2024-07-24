#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo -e "usage: $0 <chunk number>\n\t(chunk number should be 1, 2, 3a or 3b.)"
    exit
fi

chunk_num=$1
scp_path="scp/2024/train-360_part${chunk_num}.scp"
output_folder="output_data/2024/train-360"
checkpoint_dir="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--bark"
data_root="/medias/speech/projects/panariel/voice_privacy_2024/VPC24_itg"
voice_dir="suno_voices/v2"
ds_type="libri"
config_file='accelerate_config.yaml'


accelerate launch --config_file $config_file inference.py \
  "$scp_path" \
  "$output_folder" \
  "$checkpoint_dir" \
  --data_root $data_root \
  --voice_dir $voice_dir \
  --target_rate 16000 \
  --vocos_ckp "char_vocos_config.yaml,/medias/speech/projects/panariel/train_vocos/vocos/logs/lightning_logs/version_4/checkpoints/last.ckpt" \
  --ds_type $ds_type 2>&1 | tee "logs/inference_libri-360_part${chunk_num}_stdout.log"
