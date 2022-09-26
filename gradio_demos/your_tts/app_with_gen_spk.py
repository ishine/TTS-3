import os
import numpy as np

import gradio as gr

import torch
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from matplotlib import pylab


from pathlib import Path
from datetime import datetime
import subprocess
import librosa
import librosa.display

import os

from TTS.utils.audio.numpy_transforms import save_wav

torch.set_num_threads(32)

source_path = Path(__file__).resolve()
ROOT_PATH = source_path.parent

# model_path = "/data/TTS/output/YourTTS-variant17-August-11-2022_11+58AM-f6aa3e4a/checkpoint_2635000.pth"
# config_path = "/data/TTS/output/YourTTS-variant17-August-11-2022_11+58AM-f6aa3e4a/config.json"

# model_path = "/data/TTS/output/YourTTS-variant17-August-12-2022_06+01PM-af3a2b1a/checkpoint_2940000.pth"
# config_path = "/data/TTS/output/YourTTS-variant17-August-12-2022_06+01PM-af3a2b1a/config.json"

# model_path = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS_with_pitch/YourTTS_modulated/YourTTS-variant17.4c83e719498b451a9a7640e645377800/models/checkpoint_2785000.pth"
# config_path = "/data/TTS/output/YourTTS-variant17-August-12-2022_06+01PM-af3a2b1a/config.json"

# model_path = "/data/best_models/variant17.1/checkpoint_3290000.pth"
# config_path = "/data/best_models/variant17.1/model_config.json"

model_path = "/data/best_models/ecyourtts_v20_video_game/checkpoint_3595000.pth"
config_path = "/data/best_models/ecyourtts_v20_video_game/model_config.json"

language_path = None
speakers_file = os.path.join(ROOT_PATH, "models/speakers.json")

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

# pylint: disable=global-statement
model_config = load_config(config_path)
model = setup_tts_model(config=model_config)
model.load_checkpoint(model_config, model_path, eval=True, strict=False)
model.speaker_manager.init_encoder(encoder_model_path, encoder_config_path, False)
model.emotion_manager.load_embeddings_from_list_of_files([
            "/raid/datasets/emotion_embeddings/Large_Hubert/emotions_embeddings_ESD+Skyrim+VCTK+LibriTTS_with_LibriTTS_outliers/emotions_with_undefined_labels_new_keys.pth",
            "/raid/datasets/emotion_embeddings/Large_Hubert/new_keys/emotions_embeddings_Game_dataset/emotions.pth"
        ])
model.cuda()

ORIGINAL_LENGTH_SCALE = model.length_scale
SPEAKERS = model.speaker_manager.speaker_names
EMOTION_NAMES = model.emotion_manager.emotion_names

print(f" > Number of speakers: {len(SPEAKERS)}")

esd_train_config = BaseDatasetConfig(
    formatter="esd",
    meta_file_train="train",  # TODO: compute emotion and d-vectors for test and evaluation splits
    path="/raid/datasets/Emotion/ESD-44kHz-VAD/English/",
    meta_file_val="evaluation",
)

skyrim_config = BaseDatasetConfig(
    formatter="coqui",
    path="/raid/datasets/skyrim_dataset_44khz_fixed_vad",
    meta_file_train="metadata_new_fixed_without_emptly_text_and_audios_filtered.csv",
    ignored_speakers=["crdragonpriestvoice", "crdogvoice"],
)

vctk_config = BaseDatasetConfig(
    formatter="vctk",
    meta_file_train="metadata.csv",
    path="/raid/datasets/VCTK_NEW_44khz_removed_silence_silero_vad/",
)

libritts_360_config = BaseDatasetConfig(
    formatter="libri_tts",
    path="/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/",
)

libritts_100_config = BaseDatasetConfig(
    formatter="libri_tts",
    path="/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/",
)

game_config = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="video_game_dataset",
    path="/raid/datasets/video_game_dataset/video_game_dataset_bwe_44khz/",
    meta_file_train="metadata_filtered.csv",
)

train_samples, _ = load_tts_samples(
    [vctk_config, libritts_360_config, libritts_100_config, esd_train_config, skyrim_config, game_config],
    eval_split=False,
)

def speaker_name_to_wav(speaker_name):
    for sample in train_samples:
        if sample["speaker_name"] == speaker_name:
            return sample["audio_file"]
    raise RuntimeError


def create_style_wav(wav_file):
    return torch.from_numpy(librosa.load(wav_file.name, sr=model.config.audio.sample_rate)[0][None, :])


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
    subprocess.check_call(
        ["ffmpeg-normalize", wav_file, "-nt", "rms", "-t=" + str(db), "-o", wav_file, "-ar", f"{ENCODER_SR}", "-f"]
    )
    # subprocess.check_call(["ffmpeg-normalize", wav_file, '-nt', "rms", '-o', wav_file, '-ar', f"{ENCODER_SR}", '-f'])


def build_pitch_transformation(
    pitch_transform_flatten, pitch_transform_invert, pitch_transform_amplify, pitch_transform_shift
):
    # if args.pitch_transform_custom:
    #     def custom_(pitch, pitch_lens, mean, std):
    #         return (pitch_transform_custom(pitch * std + mean, pitch_lens)
    #                 - mean) / std
    #     return custom_

    fun = "pitch"
    if pitch_transform_flatten:
        fun = f"({fun}) * 0.0"
    if pitch_transform_invert:
        fun = f"({fun}) * -1.0"
    if pitch_transform_amplify != 1.0:
        ampl = pitch_transform_amplify
        fun = f"({fun}) * {ampl}"
    if pitch_transform_shift != 0.0:
        hz = pitch_transform_shift
        fun = f"({fun}) + {hz} / std"

    if fun == "pitch":
        return None

    return eval(f"lambda pitch, pitch_lens, mean, std: {fun}")


def rms_norm(*, wav: np.ndarray = None, db_level: float = -27.0, **kwargs) -> np.ndarray:
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


speaker_embedding = None


def tts(
    speaker_wav,
    uploaded_wav,
    text,
    speaker_id,
    emotion_name,
    dbfs,
    sample_gmm,
    use_last_spkemb,
    gmm_noise_scale,
    pitch_flatten,
    pitch_invert,
    pitch_amplify,
    pitch_shift,
    speed,
    model_noise_scale,
    sdp_noise_scale,
):
    global REQ_COUNTER
    global speaker_embedding

    cmd = f"rm -fr {GENERATED_WAV_PATH}/*"
    print(cmd)
    os.system(cmd)

    print(f"\n__________NEW_REQUEST {REQ_COUNTER} -({return_time()})___________")
    print(f" > Speaker wav: {speaker_wav}")
    print(f" > Emotion name: {emotion_name}")
    print(f" > Uploaded wav: {uploaded_wav}")
    print(f" > Text: {text}")
    print(f" > DBFS: {dbfs}")
    print(f" > Sample GMM: {sample_gmm}")
    print(f" > GMM noise scale: {gmm_noise_scale}")
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

    speaker_name = speaker_id
    # style_wav = speaker_name_to_wav(style_id)
    # orig_wav = speaker_name_to_wav(speaker_name)
    if speaker_wav is not None:
        wav, wav_file = process_speaker_wav(speaker_wav, dbfs)
    elif uploaded_wav is not None:
        wav, wav_file = process_speaker_wav(uploaded_wav, dbfs)

    # copy input files
    output_path = os.path.join(GENERATED_WAV_PATH, "output.wav")
    if wav_file:
        if isinstance(wav_file, list):
            for wf in wav_file:
                input_path = os.path.join(SPEAKER_WAV_PATH, os.path.basename(wf))
        else:
            input_path = os.path.join(SPEAKER_WAV_PATH, os.path.basename(wav_file))

        if isinstance(wav_file, list):
            wav_file_names = "_".join([Path(wf).stem for wf in wav_file])
            output_path = os.path.join(GENERATED_WAV_PATH, wav_file_names + ".wav")
        else:
            output_path = os.path.join(GENERATED_WAV_PATH, os.path.basename(wav_file))

    pitch_transform = build_pitch_transformation(pitch_flatten, pitch_invert, pitch_amplify, pitch_shift)


    if uploaded_wav is not None or speaker_wav is not None:
        speaker_embedding = model.speaker_manager.compute_embedding_from_clip(wav_file)
    elif not use_last_spkemb:

        # compute spk_emb
        speaker_embedding = model.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)

        # sample GMM
        if sample_gmm:
            speaker_embedding = np.random.randn(speaker_embedding.shape[0]) * gmm_noise_scale + speaker_embedding

    # compute emo_emb
    emotion_embedding = model.emotion_manager.get_mean_embedding(emotion_name)

    # run the model
    model.length_scale = ORIGINAL_LENGTH_SCALE / speed
    output_dict = model.synthesize(
        text=text,
        speaker_id=None,
        language_id=None,
        d_vector=speaker_embedding,
        ref_waveform=None,
        emotion_vector=emotion_embedding,
        emotion_id=emotion_name,
        pitch_transform=pitch_transform,
        noise_scale=model_noise_scale,
        sdp_noise_scale=sdp_noise_scale,
    )
    wav = output_dict["wav"][0]
    save_wav(wav=wav, path=output_path, sample_rate=model_config.audio["sample_rate"])
    print(f" > Output audio saved to - {output_path}")

    # plot spectrogram
    fig1, ax = pylab.subplots()
    M = librosa.feature.melspectrogram(
        y=np.array(wav), sr=model.config.audio.sample_rate, n_mels=model.config.audio.num_mels
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, ax=ax)
    ax.set(title="Mel spectrogram display")
    fig1.colorbar(img, ax=ax, format="%+2.f dB")

    # plot figures for the first sentence
    figures = model.plot_outputs(text, output_dict["wav"], output_dict["alignments"], output_dict["outputs"])
    return (
        output_path,
        # orig_wav,
        fig1,
        uploaded_wav.name if uploaded_wav else None,
        figures["alignment"],
        figures["pitch_from_audio"],
        figures["pitch_predicted"],
        figures["pitch_avg_from_audio"],
    )


def save_embedding(filename):
    global speaker_embedding
    path = f"/data/TTS/gradio_demos/your_tts/embeddings/{filename}.npy"
    print(" > Saving the embedding...")
    if os.path.exists(path):
        print(" > Embedding already exists!")
        return "Embedding with this name already exists! Try a different name."
    np.save(path, speaker_embedding)
    return "Done"


article = """
Here is a text for you to read while recording your voice for some 🪄.

Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world. He got his father’s permission to leave on an adventure, kissed his parents goodbye, mounted his black horse, and galloped away down the high road. Soon the grey towers of the old castle, in which he had been born, disappeared behind him.
"""

iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.Audio(
            source="microphone",
            label="Step 1. Record your voice. - You can say or read anything you want -",
            type="file",
        ),
        gr.File(file_count="single"),
        gr.Textbox(
            label="Step 2. Write a sentence - This is what the model reads - (Max. 500 characters)",
            value="Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world.",
        ),
        gr.Dropdown(SPEAKERS, label="Choose a speaker"),
        gr.Dropdown(EMOTION_NAMES, label="Choose emotion"),
        gr.Number(value=-27, label="Target dBFS"),
        gr.Checkbox(label="Sample new speaker with GMM"),
        gr.Checkbox(label="Use the last sampled embedding"),
        gr.Number(value=0.05, label="GMM noise scale"),
        gr.Checkbox(label="Pitch Flatten (Pitch * 0)"),
        gr.Checkbox(label="Pitch Invert (Pitch * -1)"),
        gr.Number(value=1.0, label="Pitch Amplify (Pitch * Pitch Amplify)"),
        gr.Number(value=0.0, label="Pitch Shift (Pitch + Pitch Shift in Hz)"),
        gr.Number(value=1.0, label="Speed"),
        # gr.Number(value=0.05, label="Denoise strength"),
        gr.Number(value=0.44, label="Model Noise Scale"),
        gr.Number(value=0.44, label="SDP Noise Scale"),
    ],
    outputs=[
        gr.outputs.Audio(label="Output Speech."),
        # gr.outputs.Audio(label="Original Speech."),
        gr.Plot(label="Spectrogram"),
        gr.outputs.Audio(label="Uploaded Speech."),
        gr.Plot(label="Durations"),
        gr.Plot(label="Pitch from audio"),
        gr.Plot(label="Pitch Avg from audio"),
        gr.Plot(label="Pitch Avg Pred"),
    ],
    allow_flagging=False,
    # flagging_options=['error', 'bad-quality', 'wrong-pronounciation'],
    article=article,
)


save_speaker = gr.Interface(
    fn=save_embedding,
    inputs=[gr.Textbox(label="Placeholder text", value="")],
    outputs=[gr.Textbox(label="Placeholder text", value="")],
)

tabbed_interface = gr.TabbedInterface(
    [iface, save_speaker], ["Text-to-speech", "Save Speaker"], analytics_enabled=False
)

if __name__ == "__main__":
    tabbed_interface.launch(
        share=False,
        debug=False,
        server_port=5013,
        server_name="0.0.0.0",
        enable_queue=True,
    )
