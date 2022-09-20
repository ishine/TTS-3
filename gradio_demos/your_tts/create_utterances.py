import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.config import load_config


torch.set_num_threads(32)


output_path = "/raid/datasets/tts_outputs/variant17.1_denormed_2735000/"

# variant 17
# model_path = "/data/TTS/output/YourTTS-variant17-August-12-2022_06+01PM-af3a2b1a/checkpoint_2940000.pth"
# config_path = "/data/TTS/output/YourTTS-variant17-August-12-2022_06+01PM-af3a2b1a/config.json"

# variant 16
# model_path = "/data/best_models/variant16/checkpoint_2555000.pth"
# config_path = "/data/best_models/variant16/model_config.json"

# variant 16 selected speakers
# model_path = "/data/best_models/variant16_selected_speakers/checkpoint_2580000.pth"
# config_path = "/data/best_models/variant16_selected_speakers/model_config.json"

# model_path = "/data/best_models/variant17.1/checkpoint_3290000.pth"
# config_path = "/data/best_models/variant17.1/model_config.json"

# model_path = "/data/best_models/variant17.2/checkpoint_2800000.pth"
# config_path = "/data/best_models/variant17.2/model_config.json"

model_path = "/data/best_models/ecyourtts_variant17.1_denorm/checkpoint_2735000.pth"
config_path = "/data/best_models/ecyourtts_variant17.1_denorm/model_config.json"


config = load_config(config_path)
model = Vits.init_from_config(config)
model.load_checkpoint(config, model_path, eval=True)
model.cuda()
model.inference_noise_scale = 0.66
model.inference_noise_scale_dp = 0.0


speaker_ids = [
    "LTTS_1224",
    "LTTS_1263",
    "LTTS_1349",
    "LTTS_1355",
    "LTTS_1624",
    "LTTS_1743",
    "LTTS_227",
    "LTTS_233",
    "LTTS_2436",
    "LTTS_26",
    "LTTS_2673",
    "LTTS_2893",
    "LTTS_2952",
    "LTTS_3235",
    "LTTS_3242",
    "LTTS_3436",
    "LTTS_403",
    "LTTS_4214",
    "LTTS_4267",
    "LTTS_4340",
    "LTTS_4406",
    "LTTS_4680",
    "LTTS_4830",
    "LTTS_5022",
    "LTTS_511",
    "LTTS_5393",
    "LTTS_5652",
    "LTTS_5808",
    "LTTS_6000",
    "LTTS_6064",
    "LTTS_6078",
    "LTTS_6081",
    "LTTS_6367",
    "LTTS_7067",
    "LTTS_707",
    "LTTS_7447",
    "LTTS_83",
    "LTTS_850",
]

transcripts = {
    "Neutral": "Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.",
    "Anger": "I cannot express in words how angry I am at this moment. Please refrain from talking with me for now.",
    "Happy": "I'm trilled to share this news with everyone. It's a honor to be in this project.",
    "Sad": "It's a really sad story but, if you try to restrain your tears I'll tell you about it.",
    "Surprise": "I can't believe you can able to do this on time. I'm so happy to see you.",
    "Undefined": "Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.",
}

outputs = {}
for speaker_id in speaker_ids:
    speaker_outputs = {}
    for emotion, transcript in transcripts.items():
        output = model.synthesize(text=transcript, speaker_id=speaker_id, emotion_id=emotion)
        speaker_outputs[emotion] = output["wav"]
    outputs[speaker_id] = speaker_outputs

# create outputs folder
import os
from TTS.utils.audio.numpy_transforms import save_wav

for speaker_name, outs in outputs.items():
    speaker_path = os.path.join(output_path, speaker_name)
    os.makedirs(speaker_path, exist_ok=True)
    for emotion_name, wav in outs.items():
        file_name = f"{emotion_name}.wav"
        wav_path = os.path.join(speaker_path, file_name)
        save_wav(wav=wav.squeeze(0), path=wav_path, sample_rate=model.config.audio.sample_rate)