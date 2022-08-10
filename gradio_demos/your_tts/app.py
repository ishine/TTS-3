from bdb import Breakpoint
import os
import numpy as np

import gradio as gr

import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from matplotlib import pylab


from pathlib import Path
from datetime import datetime
import subprocess
import librosa
import librosa.display

import os

torch.set_num_threads(32)

source_path = Path(__file__).resolve()
ROOT_PATH = source_path.parent

# model_path = "/data/TTS/output/YourTTS-variant16-emotion-August-06-2022_10+25PM-409b528f/checkpoint_2550000.pth"
# config_path = "/data/TTS/output/YourTTS-variant16-emotion-August-06-2022_10+25PM-409b528f/config.json"

# model_path = "/data/TTS/output/YourTTS-variant16-emotion-August-09-2022_02+25PM-0bd810f8/checkpoint_2590000.pth"
# config_path = "/data/TTS/output/YourTTS-variant16-emotion-August-09-2022_02+25PM-0bd810f8/config.json"

model_path = "/data/TTS/output/YourTTS-variant16-emotion-August-09-2022_12+45AM-409b528f/checkpoint_2585000.pth"
config_path = "/data/TTS/output/YourTTS-variant16-emotion-August-09-2022_12+45AM-409b528f/config.json"

language_path = None
speakers_file = os.path.join(ROOT_PATH, 'models/speakers.json')

encoder_model_path = os.path.join(ROOT_PATH, "models/model_se.pth.tar")
encoder_config_path = os.path.join(ROOT_PATH, "models/config_se.json")


SPEAKER_WAV_PATH = os.path.join(ROOT_PATH, "stored_data/submitted_wavs")
GENERATED_WAV_PATH = os.path.join(ROOT_PATH, "stored_data/generated_wavs")
FLAGGED_PATH = os.path.join(ROOT_PATH, "stored_data/flagged")

ENCODER_SR = 16_000
TRAGET_DBFS = -27

ERROR1 = os.path.join(ROOT_PATH, "data/no_record_error.wav")  # no voice recording
ERROR2 = os.path.join(ROOT_PATH, "data/no_input_sentence.wav")  # no input sentence

os.makedirs(SPEAKER_WAV_PATH, exist_ok=True)
os.makedirs(GENERATED_WAV_PATH, exist_ok=True)

REQ_COUNTER = 0

gr.close_all()

synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=speakers_file,
        tts_languages_file=language_path,
        vocoder_checkpoint=None,
        vocoder_config = None,
        encoder_checkpoint= encoder_model_path,
        encoder_config= encoder_config_path,
        use_cuda=False
    )

ORIGINAL_LENGTH_SCALE = synthesizer.tts_model.length_scale
SPEAKERS = synthesizer.tts_model.speaker_manager.speaker_names
EMOTION_EMB_PATH = "/raid/datasets/emotion_embeddings/Large_Hubert/emotions_embeddings_ESD_with_GT_labels/emotions.pth"

emotion_embeddings = torch.load(EMOTION_EMB_PATH)
EMOTION_NAMES = synthesizer.tts_model.emotion_manager.emotion_names

esd_train_config = BaseDatasetConfig(
    name="esd",
    meta_file_train="train",  # TODO: compute emotion and d-vectors for test and evaluation splits
    path="/raid/datasets/Emotion/ESD-44kHz-VAD/English/",
    meta_file_val="evaluation"
)

skyrim_config = BaseDatasetConfig(
    name="coqui",
    path="/raid/datasets/skyrim_dataset_44khz_fixed_vad",
    meta_file_train = "metadata_new_fixed_without_emptly_text_and_audios_filtered.csv",
    ignored_speakers = ["crdragonpriestvoice", "crdogvoice"],
)

vctk_config = BaseDatasetConfig(
    name="vctk",
    meta_file_train="metadata.csv",
    path="/raid/datasets/VCTK_NEW_44khz_removed_silence_silero_vad/",
)

libritts_360_config = BaseDatasetConfig(
    name="libri_tts",
    path="/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/",
)

libritts_100_config = BaseDatasetConfig(
    name="libri_tts",
    path="/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/",
)

train_samples, _ = load_tts_samples(
    [vctk_config, libritts_360_config, libritts_100_config, esd_train_config, skyrim_config],
    eval_split=False,
)


def speaker_name_to_wav(speaker_name):
    for sample in train_samples:
        if sample["speaker_name"] == speaker_name:
            return sample["audio_file"]
    raise RuntimeError


def create_style_wav(wav_file):
    return torch.from_numpy(librosa.load(wav_file.name, sr=synthesizer.tts_model.config.audio.sample_rate)[0][None, :])


def process_speaker_wav(speake_wav, dbfs):
    process_with_ffmpeg(speake_wav.name, dbfs)
    return None, speake_wav.name


def process_uploaded_wav(wav_file, dbfs):
    process_with_ffmpeg(wav_file.name, dbfs)
    return None, wav_file.name


def process_uploaded_files(wav_files, dbfs):
    wavs = []
    saved_files = []
    for wav_file in wav_files:
        wav, saved_file = process_uploaded_wav(wav_file, dbfs)
        wavs += [wav]
        saved_files += [saved_file]
    return wavs, saved_files


def return_time():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def process_with_ffmpeg(wav_file, db=-27):
    # ffmpeg-normalize "$outFile" -nt rms -t=-27  -o "$outFile" -ar 16000 -f
    subprocess.check_call(["ffmpeg-normalize", wav_file, '-nt', "rms", "-t="+str(db), '-o', wav_file, '-ar', f"{ENCODER_SR}", '-f'])
    # subprocess.check_call(["ffmpeg-normalize", wav_file, '-nt', "rms", '-o', wav_file, '-ar', f"{ENCODER_SR}", '-f'])


def build_pitch_transformation(pitch_transform_flatten, pitch_transform_invert, pitch_transform_amplify, pitch_transform_shift):
    # if args.pitch_transform_custom:
    #     def custom_(pitch, pitch_lens, mean, std):
    #         return (pitch_transform_custom(pitch * std + mean, pitch_lens)
    #                 - mean) / std
    #     return custom_

    fun = 'pitch'
    if pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if pitch_transform_amplify != 1.0:
        ampl = pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if pitch_transform_shift != 0.0:
        hz = pitch_transform_shift
        fun = f'({fun}) + {hz} / std'

    if fun == 'pitch':
        return None

    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


def rms_norm(*, wav: np.ndarray = None, db_level: float = -27.0, **kwargs) -> np.ndarray:
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


def tts(speaker_wav, uploaded_wav, text, speaker_id, emotion_name, dbfs, pitch_flatten, pitch_invert, pitch_amplify, pitch_shift, speed, model_noise_scale):
    global REQ_COUNTER

    cmd = f"rm -fr {GENERATED_WAV_PATH}/*"
    print(cmd)
    os.system(cmd)

    print(f"\n__________NEW_REQUEST {REQ_COUNTER} -({return_time()})___________")
    print(f" > Speaker wav: {speaker_wav}")
    print(f" > Emotion name: {emotion_name}")
    print(f" > Uploaded wav: {uploaded_wav}")
    print(f" > Text: {text}")
    print(f" > DBFS: {dbfs}")
    print(f" > Pitch Flatten: {pitch_flatten}")
    print(f" > Pitch Invert: {pitch_invert}")
    print(f" > Pitch Amplify: {pitch_amplify}")
    print(f" > Pitch Shift: {pitch_shift}")
    print(f" > Speed: {speed}")
    print(f" > Running inference...")
    REQ_COUNTER += 1

    lang = None
    speaker_name = None
    wav_file = None
    # dbfs = -27

    # if uploaded_wav is not None:
    #     if isinstance(uploaded_wav, list):
    #         wav, wav_file = process_uploaded_files(uploaded_wav, dbfs)
    #         style_wav = uploaded_wav.name
    #     else:
    #         wav, wav_file = process_uploaded_wav(uploaded_wav, dbfs)
    #         style_wav = uploaded_wav.name
    # elif speaker_wav is not None:
    #     wav, wav_file = process_speaker_wav(speaker_wav, dbfs)
    #     style_wav = speaker_wav.name
    # elif speaker_id is not None:
    speaker_name = speaker_id
    # style_wav = speaker_name_to_wav(style_id)
    orig_wav = speaker_name_to_wav(speaker_name)

    # copy input files
    output_path = os.path.join(GENERATED_WAV_PATH, "output.wav")
    if wav_file:
        if isinstance(wav_file, list):
            for wf in wav_file:
                input_path = os.path.join(SPEAKER_WAV_PATH, os.path.basename(wf))
                # copyfile(wf, input_path)
                # upload_inputs_to_s3(input_path)
                # print(f" > Input (uploaded) audio saved to - {input_path}")

        else:
            input_path = os.path.join(SPEAKER_WAV_PATH, os.path.basename(wav_file))
            # copyfile(wav_file, input_path)
            # upload_inputs_to_s3(input_path)
            # print(f" > Input (recorded) audio saved to - {input_path}")

        if isinstance(wav_file, list):
            wav_file_names = "_".join([Path(wf).stem for wf in wav_file])
            output_path = os.path.join(GENERATED_WAV_PATH, wav_file_names + ".wav")
        else:
            output_path = os.path.join(GENERATED_WAV_PATH, os.path.basename(wav_file))

    pitch_transform = build_pitch_transformation(pitch_flatten, pitch_invert, pitch_amplify, pitch_shift)

    # run TTS
    synthesizer.tts_model.length_scale = ORIGINAL_LENGTH_SCALE / speed
    wavs, outputss = synthesizer.tts(text=text, speaker_name=speaker_name, language_name=lang, speaker_wav=wav_file, emotion_name=emotion_name, pitch_transform=pitch_transform, noise_scale=model_noise_scale)
    wavs = list(rms_norm(wav=np.array(wavs)))
    synthesizer.save_wav(wavs, output_path)
    print(f" > Output audio saved to - {output_path}")
     # compute mean basis vector attn
    fig1 = None
    if "basis_attn" in outputss[0]["outputs"] and outputss[0]["outputs"]["basis_attn"] is not None:
        basis_attn = np.zeros([len(outputss), outputss[0]["outputs"]["basis_attn"].shape[1]])
        for idx, outputs in enumerate(outputss):
            basis_attn[idx, :] = outputs["outputs"]["basis_attn"][0].cpu().numpy()
        basis_attn = basis_attn.mean(axis=0, keepdims=True)
        fig1 = pylab.figure()
        pylab.stem(range(basis_attn.shape[1]), basis_attn[0])

    # plot spectrogram
    fig2, ax = pylab.subplots()
    M = librosa.feature.melspectrogram(y=np.array(wavs), sr=synthesizer.tts_model.config.audio.sample_rate, n_mels=synthesizer.tts_model.config.audio.num_mels)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, ax=ax)
    ax.set(title='Mel spectrogram display')
    fig2.colorbar(img, ax=ax, format="%+2.f dB")

     # plot figures for the first sentence
    figures = synthesizer.tts_model.plot_outputs(outputss[0]["sentence"], outputss[0]["wav"], outputss[0]["alignments"], outputss[0]["outputs"])
    return output_path, orig_wav, fig1, fig2, uploaded_wav.name if uploaded_wav else None, figures["alignment"], figures["pitch_from_audio"], figures["pitch_predicted"], figures["pitch_avg_from_audio"]


article= """
Here is a text for you to read while recording your voice for some 🪄.

Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.
"""

iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.Audio(source="microphone", label="Step 1. Record your voice. - You can say or read anything you want -", type="file"),
        gr.File(file_count="single"),
        # gr.File(file_count="multiple", label="Alternative Step 1. Input Audio files. - You can upload one or more audio files - ", type="file", optional=True),
        gr.Textbox(
            label="Step 2. Write a sentence - This is what the model reads - (Max. 500 characters)",
            value="Erase this text and enter your own.",
        ),
        gr.Dropdown(SPEAKERS, label="Choose a speaker"),
        gr.Dropdown(EMOTION_NAMES, label="Choose emotion"),
        gr.Number(value=-27, label="Target dBFS"),
        # gr.Number(value=0.22, label="Duraiton Noise Scale"),
        # gr.Number(value=0.22, label="Model Noise Scale")
        gr.Checkbox(label="Pitch Flatten (Pitch * 0)"),
        gr.Checkbox(label="Pitch Invert (Pitch * -1)"),
        gr.Number(value=1.0, label="Pitch Amplify (Pitch * Pitch Amplify)"),
        gr.Number(value=0.0, label="Pitch Shift (Pitch + Pitch Shift in Hz)"),
        gr.Number(value=1.0, label="Speed"),
        # gr.Number(value=0.11, label="Duration Noise Scale"),
        gr.Number(value=0.66, label="Model Noise Scale"),
    ],
    outputs=[gr.outputs.Audio(label="Output Speech."), gr.outputs.Audio(label="Original Speech."), gr.Plot(label="Basis attention weights"), gr.Plot(label="Spectrogram"), gr.outputs.Audio(label="Uploaded Speech."), gr.Plot(label="Durations"), gr.Plot(label="Pitch from audio"), gr.Plot(label="Pitch Avg from audio"), gr.Plot(label="Pitch Avg Pred")],
    allow_flagging=False,
    # flagging_options=['error', 'bad-quality', 'wrong-pronounciation'],
    layout="vertical",
    # flagging_dir=FLAGGED_PATH,
    # server_port=7860,
    # server_name="0.0.0.0",
    allow_screenshot=False,
    enable_queue=False,
    article=article,
)
iface.launch(share=False, debug=False, server_port=5007, server_name="0.0.0.0",)
