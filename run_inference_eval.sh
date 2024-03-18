#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo -e "usage: $0 <ds_type> {enrolls,trials}\n\t(where 'ds_type' should be {libri,vtck}_{dev,test})"
    exit
fi

ds_name=$1
partition=$2

scp_path="scp/${ds_name}_${partition}_wav.scp"
output_folder="output_data/${ds_name}_${partition}"
checkpoint_dir="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--bark"
data_root="Voice-Privacy-Challenge-2022/baseline"
voice_dir="suno_voices/v2"
mapping_file="speaker_mappings/speaker_mapping_${ds_name}_${partition}.json"
ds_type=${ds_name%_*}
config_file='accelerate_config.yaml'


accelerate launch --config_file $config_file inference.py \
  "$scp_path" \
  "$output_folder" \
  $checkpoint_dir \
  --data_root $data_root \
  --voice_dir $voice_dir \
  --mapping_file "$mapping_file" \
  --speaker_level \
  --ds_type $ds_type 2>&1 | tee inference_${ds_name}_stdout.log

  # the 'speaker_level' parameter is technically not needed, but whatever