import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.config import load_config


torch.set_num_threads(32)


output_path = "/raid/datasets/tts_outputs/ecyourtts_v20_video_game_checkpoint_3685000/"

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

model_path = "/data/best_models/ecyourtts_v20_video_game/checkpoint_3685000.pth"
config_path = "/data/best_models/ecyourtts_v20_video_game/model_config.json"


config = load_config(config_path)
model = Vits.init_from_config(config)
model.load_checkpoint(config, model_path, eval=True)
model.cuda()
model.inference_noise_scale = 0.44
model.inference_noise_scale_dp = 0.0
denoise_strength = 0.05

# load emb files
speakers_file = ["/raid/datasets/speaker_embeddings_new_key/speakers_vctk_libritts_esd_skyrim.pth", "/data/TTS/gradio_demos/your_tts/embeddings/dvectors.pth", "/data/TTS/gradio_demos/your_tts/dvector.pth"]
model.speaker_manager.load_embeddings_from_list_of_files(speakers_file)


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
    "adult_male_calm",
    "adult_male_rough",
    "female_adult_excited",
    "female_adult_expressive",
    "female_adult_rough",
    "female_adult_soft",
    "female_child",
    "female_child_rough",
    "female_elder1",
    "female_elder2",
    "female_elder3",
    "female_elder4",
    "female_elder5",
    "female_elder6",
    "female_elder_eager_dynamic",
    "female_elder_eager_dynamic2",
    "female_elder_rough",
    "female_elder_soft",
    "male_adult_accented",
    "male_adult_accented_dwarf",
    "male_adult_distinct",
    "male_adult_sneaky",
    "male_adult_soft_polite",
    "male_child",
    "male_child2",
    "male_child_expressive",
    "male_elder1",
    "male_elder2",
    "male_elder3",
    "male_elder4",
    "male_elder_dynamic",
    "elder_18_m",
]

transcripts = {
    "Neutral": "Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.",
    "Neutral": "I'm Commander Shepard, and this is my favorite store on the Citadel!",
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
        output = model.synthesize(text=transcript, speaker_id=speaker_id, emotion_id=emotion, denoise_strength=denoise_strength)
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
