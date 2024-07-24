#!/bin/bash

scp_path="/medias/speech/projects/panariel/train_vocos/data/libri_dev/wav_enrolls.scp"
output_folder="output_data/libri_dev_anon_vocos_2"
checkpoint_dir="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--bark"
#vocos_ckp="/medias/speech/projects/panariel/train_vocos/vocos/logs/lightning_logs/version_3/checkpoints/last.ckpt"
vocos_ckp="char_vocos_config.yaml,/medias/speech/projects/panariel/train_vocos/vocos/logs/lightning_logs/version_4/checkpoints/last.ckpt"
data_root="/medias/speech/projects/panariel/train_vocos"
voice_dir="suno_voices/v2"
ds_type="libri"
config_file='accelerate_config.yaml'


accelerate launch --config_file $config_file inference.py \
  "$scp_path" \
  "$output_folder" \
  "$checkpoint_dir" \
  --vocos_ckp $vocos_ckp \
  --data_root $data_root \
  --voice_dir $voice_dir \
  --ds_type $ds_type 2>&1 | tee inference_libri_dev_CharVocos_stdout.log