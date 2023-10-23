#!/usr/bin/env python3`
import glob
import os
import shutil

import torch

from tests import get_tests_data_path, get_tests_output_path, run_cli
from TTS.utils.generic_utils import get_user_data_dir

def test_xtts():
    """XTTS is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )
    else:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )

def test_xtts_v1_1():
    """XTTS is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1.1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )
    else:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1.1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )

def test_xtts_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v1")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1

def test_xtts_v1_1_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v1.1")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1

