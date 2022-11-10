import numpy as np
import torch
from TTS.tts.models.vits import Vits
from TTS.config import load_config
import soundfile as sf
import os
from tqdm import tqdm


torch.set_num_threads(32)


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save. Shape (n_values,).
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav = wav / wav.max() * 0.7326078
    sf.write(path, wav, sample_rate, **kwargs)

##### -----> SET THESE PARAMETERS ❗️❗️❗️

model_path = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS_with_pitch/Variant21/ECYourTTS_coqui_studio/ECYourTTS-v21.1-ext_enc.a3125c78f4494b288afe450d851ce233/models/checkpoint_5455000.pth"
config_path = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS_with_pitch/Variant21/ECYourTTS_coqui_studio/ECYourTTS-v21.1-ext_enc.a3125c78f4494b288afe450d851ce233/artifacts/model_config/model_config.json"
root_path = "/raid/datasets/tts_outputs/v21.3_ext_enc_e2e/"

##### -----> DENOISER PARAMETERS ❗️❗️❗️

denoiser_mode = "normal"
denoise_strength = [0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
denoise_spk_dep = False

##### -----> MODEL SET UP

config = load_config(config_path)
model = Vits.init_from_config(config)
model.load_checkpoint(config, model_path, eval=True, strict=False)
model.cuda()
model.inference_noise_scale = 0.44
model.inference_noise_scale_dp = 0.0

# load emb files
speakers_file = ["/data/TTS/gradio_demos/your_tts/embeddings/speakers_v2.pth"]
model.speaker_manager.load_embeddings_from_list_of_files(speakers_file)
model.emotion_manager.load_embeddings_from_file("/data/TTS/gradio_demos/your_tts/embeddings/emotions_v1.pth")

transcripts = {
    "Neutral": "Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.",
    "Neutral": "I'm Commander Shepard, and this is my favorite store on the Citadel!",
    "Anger": "I cannot express in words how angry I am at this moment. Please refrain from talking with me for now.",
    "Happy": "I'm trilled to share this news with everyone. It's a honor to be in this project.",
    "Sad": "It's a really sad story but, if you try to restrain your tears I'll tell you about it.",
    "Surprise": "I can't believe you can able to do this on time. I'm so happy to see you.",
}

for ds in denoise_strength:
    output_path = os.path.join(root_path, f"{denoiser_mode}_{denoise_spk_dep}_{ds}_denoised/")
    outputs = {}
    for speaker_id in tqdm(model.speaker_manager.speaker_names):
        speaker_outputs = {}
        for emotion, transcript in transcripts.items():
            output = model.synthesize(
                text=transcript,
                speaker_id=speaker_id,
                emotion_id=emotion,
                denoise_strength=ds,
                denoiser_mode=denoiser_mode,
                denoiser_spk_dep=denoise_spk_dep,
            )
            speaker_outputs[emotion] = output["wav"]
        outputs[speaker_id] = speaker_outputs

    # create outputs folder
    import os

    for speaker_name, outs in outputs.items():
        speaker_path = os.path.join(output_path, speaker_name)
        os.makedirs(speaker_path, exist_ok=True)
        for emotion_name, wav in outs.items():
            file_name = f"{emotion_name}.wav"
            wav_path = os.path.join(speaker_path, file_name)
            save_wav(wav=wav.squeeze(0), path=wav_path, sample_rate=model.config.audio.sample_rate)
