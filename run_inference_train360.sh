#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo -e "usage: $0 <chunk number>\n\t(chunk number should be 1, 2, 3 or 4.)"
    exit
fi

chunk_num=$1
scp_path="scp/libri360_part${chunk_num}.scp"
output_folder="output_data/libri360"
checkpoint_dir="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--bark"
data_root="Voice-Privacy-Challenge-2022/baseline"
voice_dir="suno_voices/v2"
ds_type="libri"
config_file='accelerate_config.yaml'


accelerate launch --config_file $config_file inference.py \
  "$scp_path" \
  "$output_folder" \
  $checkpoint_dir \
  --data_root $data_root \
  --voice_dir $voice_dir \
  --ds_type $ds_type 2>&1 | tee "inference_libri360_part${chunk_num}_stdout.log"
