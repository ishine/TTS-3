#!/bin/bash

if [ "$#" -ne 4 ]; then
    # echo -e "usage: $0 <ds_type> {enrolls,trials}\n\t(where 'ds_type' should be {libri,vtck}_{dev,test})"
    echo -e "usage: $0 {libri,iemocap} {dev,test} {enrolls,trials_m,trials_f> <data_root>\n\t(if dataset is iemocap, 'trials' or 'enrolls' is ignored)"
    exit
fi

ds_name=$1
partition=$2
tr_en=$3
data_root=$4

if [[ "$ds_name" == "libri" ]]; then
  scp_path="scp/2024/${ds_name}_${partition}_${tr_en}.scp"
  output_folder="output_data/2024/${ds_name}_${partition}_${tr_en}"
  log_file="inference_${ds_name}_${partition}_${tr_en}_stdout.log"
else
  scp_path="scp/2024/${ds_name}_${partition}.scp"
  output_folder="output_data/2024/${ds_name}_${partition}"
  log_file="inference_${ds_name}_${partition}_stdout.log"
fi


checkpoint_dir="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--bark"
#data_root="Voice-Privacy-Challenge-2022/baseline"
voice_dir="suno_voices/v2"
config_file='accelerate_config.yaml'


accelerate launch --config_file $config_file inference.py \
  "$scp_path" \
  "$output_folder" \
  "$checkpoint_dir" \
  --data_root "$data_root" \
  --voice_dir $voice_dir \
  --target_rate 16000 \
  --vocos_ckp "char_vocos_config.yaml,/medias/speech/projects/panariel/train_vocos/vocos/logs/lightning_logs/version_4/checkpoints/last.ckpt" \
  --ds_type "$ds_name" 2>&1 | tee "$log_file"