import os
import random
import numpy as np
import gradio as gr
import torch
import soundfile as sf
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.helpers import average_over_durations
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.tts.models.ecyourtts import EcyourTTS
from gradio_app_utils import controller_reader, controller_writer
from matplotlib import pylab
from pathlib import Path
from datetime import datetime
import subprocess
import librosa
import librosa.display

torch.set_num_threads(12)

source_path = Path(__file__).resolve()
ROOT_PATH = source_path.parent

# v21.3voc
model_path = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS_with_pitch/Variant21/ECYourTTS_coqui_studio/ECYourTTS-v21.3_e2e_voc.5a28ac1739434e7c83680a4b404a41b2/models/checkpoint_4320000.pth"
config_path = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS_with_pitch/Variant21/ECYourTTS_coqui_studio/ECYourTTS-v21.3_e2e_voc.5a28ac1739434e7c83680a4b404a41b2/artifacts/model_config/model_config.json"
port = 5030

language_path = None

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
model_config = EcyourTTS.load_config(config_path)
model = EcyourTTS.init_from_config(model_config, verbose=True)
model.load_checkpoint(model_config, model_path, eval=True, strict=False)
model.speaker_manager.init_encoder(encoder_model_path, encoder_config_path, False)
# model.cuda()

# reading custom set of speakers
speakers_file = [
    "/raid/datasets/speaker_embeddings_new_key/speakers_vctk_libritts_esd_skyrim.pth",
    "/data/TTS/gradio_demos/speakers_v1.pth",
    "/raid/datasets/video_game_dataset/speakers.pth",
]
speakers_file = ["/data/TTS/gradio_demos/ecyourtts/embeddings/speakers_studio.pth"]

model.speaker_manager.load_embeddings_from_list_of_files(speakers_file)

ORIGINAL_LENGTH_SCALE = model.length_scale
SPEAKERS = model.speaker_manager.speaker_names
EMOTION_EMB_PATH = "/raid/datasets/emotion_embeddings/Large_Hubert/emotions_embeddings_ESD_with_GT_labels/emotions.pth"

# SPEAKERS += selected_spks

emotion_embeddings = torch.load(EMOTION_EMB_PATH)
EMOTION_NAMES = model.emotion_manager.emotion_names

from sklearn.preprocessing import normalize

emotion_embeddings = {}
for en1 in EMOTION_NAMES:
    emo_emb = model.emotion_manager.get_mean_embedding(en1)
    emo_emb_res = emo_emb
    for en2 in [e for e in EMOTION_NAMES if e != en1]:
        embs2 = model.emotion_manager.get_mean_embedding(en2)
        emo_emb_res = emo_emb_res - embs2
    emotion_embeddings[en1] = normalize((emo_emb + 0.2 * emo_emb_res)[None, :], norm="l2")[0]
torch.save(emotion_embeddings, "/data/TTS/gradio_demos/ecyourtts/emotion_vecs_vector_algebra.pth")

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


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save. Shape (n_values,).
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    sf.write(path, wav, sample_rate, **kwargs)


def speaker_name_to_wav(speaker_name):
    samples = []
    for sample in train_samples:
        if sample["speaker_name"] == speaker_name:
            samples.append(sample["audio_file"])
    if len(samples) == 0:
        return None
    return random.choice(samples)


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
    subprocess.check_call(
        ["ffmpeg-normalize", wav_file, "-nt", "rms", "-t=" + str(db), "-o", wav_file, "-ar", f"{ENCODER_SR}", "-f"]
    )


def build_pitch_transformation(
    pitch_transform_flatten, pitch_transform_invert, pitch_transform_amplify, pitch_transform_shift
):

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
        fun = f"({fun}) + {hz}"

    if fun == "pitch":
        return None

    return eval(f"lambda pitch, pitch_lens, mean, std: {fun}")


def rms_norm(*, wav: np.ndarray = None, db_level: float = -27.0, **kwargs) -> np.ndarray:
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


# needed for morping a speaker
speaker_embedding = None


def phoneme_to_frame_vals(values, durations):
    avg_values = average_over_durations(values, durations)
    return avg_values


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
    pitch_control_input,
    energy_control_input,
    speed,
    duration_control_input,
    denoiser_strength,
    denoiser_mode,
    model_noise_scale,
    split_batching,
    state_vars
):

    global REQ_COUNTER
    global speaker_embedding

    # set session states
    if len(state_vars) == 0:
        state_vars["pitch_vals"] = None
        state_vars["energy_vals"] = None
        state_vars["avg_pitch_vals"] = None
        state_vars["avg_energy_vals"] = None
    else:
        pitch_vals = state_vars["pitch_vals"]
        energy_vals = state_vars["energy_vals"]
        avg_pitch_vals = state_vars["avg_pitch_vals"]
        avg_energy_vals = state_vars["avg_energy_vals"]

    cmd = f"rm -fr {GENERATED_WAV_PATH}/*"
    print(cmd)
    os.system(cmd)

    print(f"\n__________NEW_REQUEST {REQ_COUNTER} -({return_time()})___________")
    print(f" > Port: {port}")
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
    orig_wav = speaker_name_to_wav(speaker_name)

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

    if not use_last_spkemb:
        if speaker_wav:
            _, wav_file = process_speaker_wav(speaker_wav, dbfs)
            speaker_embedding = model.speaker_manager.compute_embedding_from_clip(wav_file)
        elif uploaded_wav:
            orig_wav, wav_file = process_speaker_wav(uploaded_wav, dbfs)
            speaker_embedding = model.speaker_manager.compute_embedding_from_clip(wav_file)
        else:
            # compute spk_emb
            speaker_embedding = model.speaker_manager.get_mean_embedding(
                speaker_name, num_samples=None, randomize=False
            )
            # sample speaker embedding
            if sample_gmm:
                speaker_embedding = np.random.randn(speaker_embedding.shape[0]) * gmm_noise_scale + speaker_embedding

    # compute emo_emb
    emotion_embedding = emotion_embeddings[emotion_name]

    # parse in control values
    pitch_values = None
    energy_values = None
    duration_values = None


    if duration_control_input is not None and len(duration_control_input) > 0:
        duration_values = controller_reader(duration_control_input)
        duration_values = [t[1] for t in duration_values]
        duration_values = torch.FloatTensor(duration_values)[None, None, :].to(model.device)


    if pitch_control_input is not None and len(pitch_control_input) > 0:
        avg_pitch_inp_vals = controller_reader(pitch_control_input)
        avg_pitch_inp_vals = [t[1] for t in avg_pitch_inp_vals]
        avg_pitch_inp_vals = torch.FloatTensor(avg_pitch_inp_vals)[None, None, :].to(model.device)
        avg_pitch_vals[avg_pitch_vals == 0] = avg_pitch_inp_vals[avg_pitch_vals == 0]
        pitch_mask = avg_pitch_inp_vals / avg_pitch_vals
        pitch_mask.nan_to_num_()
        pitch_mask = torch.repeat_interleave(pitch_mask, duration_values.flatten().int(), dim=2)
        pitch_values = pitch_vals * pitch_mask

    if energy_control_input is not None and len(energy_control_input) > 0:
        avg_energy_inp_vals = controller_reader(energy_control_input)
        avg_energy_inp_vals = [t[1] for t in avg_energy_inp_vals]
        avg_energy_inp_vals = torch.FloatTensor(avg_energy_inp_vals)[None, None, :].to(model.device)
        avg_energy_vals[avg_energy_vals == 0] = avg_energy_inp_vals[avg_energy_vals == 0]
        energy_mask = avg_energy_inp_vals / avg_energy_vals
        energy_mask.nan_to_num_()
        energy_mask = torch.repeat_interleave(energy_mask, duration_values.flatten().int(), dim=2)
        energy_values = energy_vals * energy_mask


    # run the model
    model.length_scale = ORIGINAL_LENGTH_SCALE / speed
    output_dict = model.synthesize(
        text=text,
        speaker_id=None,
        language_id=None,
        d_vector=speaker_embedding,
        ref_waveform=None,
        emotion_vector=emotion_embedding,
        emotion_id=None,
        pitch_transform=pitch_transform,
        noise_scale=model_noise_scale,
        sdp_noise_scale=0.0,
        denoise_strength=denoiser_strength,
        denoiser_mode=denoiser_mode,
        pitch_values=pitch_values,
        energy_values=energy_values,
        duration_values=duration_values,
        split_batch_sentences=False,
    )
    wav = output_dict["wav"][0]
    save_wav(wav=wav, path=output_path, sample_rate=model_config.audio["sample_rate"])
    print(f" > Output audio saved to - {output_path}")

    # parse out control values
    phonemes = output_dict["input_phonemes"]
    pitch_vals = output_dict["outputs"]["pitch"]
    energy_vals = output_dict["outputs"]["energy"]
    dur_vals = output_dict["outputs"]["durations"]
    avg_pitch_vals = average_over_durations(pitch_vals, dur_vals[0].cpu())
    avg_energy_vals = average_over_durations(energy_vals, dur_vals[0].cpu())
    pitch_control_data = controller_writer(phonemes[0], list(avg_pitch_vals[0].cpu().numpy()[0]))
    energy_control_data = controller_writer(phonemes[0], list(avg_energy_vals[0].cpu().numpy()[0]))
    dur_control_data = controller_writer(phonemes[0], list(dur_vals[0].cpu().numpy()[0]))

    # plot spectrogram
    fig1, ax = pylab.subplots()
    M = librosa.feature.melspectrogram(
        y=np.array(wav), sr=model.config.audio.sample_rate, n_mels=model.config.audio.num_mels
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, ax=ax, x_axis="time", y_axis="linear")
    ax.set(title="Mel spectrogram display")
    fig1.colorbar(img, ax=ax, format="%+2.f dB")

    # plot original speaker spectrogram
    fig2 = None
    if orig_wav is not None:
        orig_waveform, _ = librosa.load(orig_wav, sr=model_config.audio["sample_rate"])
        fig2, ax = pylab.subplots()
        M = librosa.feature.melspectrogram(
            y=orig_waveform, sr=model.config.audio.sample_rate, n_mels=model.config.audio.num_mels
        )
        M_db = librosa.power_to_db(M, ref=np.max)
        img = librosa.display.specshow(
            M_db,
            ax=ax,
            x_axis="time",
            y_axis="linear",
        )
        ax.set(title="Org. mel spectrogram display")
        fig2.colorbar(img, ax=ax, format="%+2.f dB")

    # plot figures for the first sentence
    figures = model.plot_outputs(text, output_dict["wav"], output_dict["outputs"]["alignments"], output_dict["outputs"])

    # update session states
    state_vars["pitch_vals"] = pitch_vals
    state_vars["energy_vals"] = energy_vals
    state_vars["dur_vals"] = dur_vals
    state_vars["avg_pitch_vals"] = avg_pitch_vals
    state_vars["avg_energy_vals"] = avg_energy_vals

    return (
        output_path,
        orig_wav,
        pitch_control_data,
        energy_control_data,
        dur_control_data,
        fig1,
        fig2,
        uploaded_wav.name if uploaded_wav else None,
        figures["alignment"],
        figures["pitch_from_audio"],
        figures["pitch_predicted"],
        figures["pitch_avg_from_audio"],
        state_vars
    )


def save_embedding(filename):
    global speaker_embedding
    path = f"/data/TTS/gradio_demos/ecyourtts/embeddings/{filename}.npy"
    print(" > Saving the embedding...")
    if os.path.exists(path):
        print(" > Embedding already exists!")
        return "Embedding with this name already exists! Try a different name."
    np.save(path, speaker_embedding)
    return "Done"


def clear_controls():
    return None, None, None


with gr.Blocks() as demo:
    state_vars = gr.State({})
    with gr.Row():
        with gr.Column() as col1:
            # Input Components
            mic_audio = gr.Audio(
                source="microphone",
                label="Step 1. Record your voice. - You can say or read anything you want -",
                type="file",
            )
            upload_file = gr.File(file_count="single")
            sentence_tb = gr.Textbox(
                label="Step 2. Write a sentence - This is what the model reads - (Max. 500 characters)",
                value="Once upon a time, the King’s youngest son became filled with the desire to go abroad, and see the world.",
            )
            speakers_dp = gr.Dropdown(SPEAKERS, value=SPEAKERS[0], label="Choose a speaker")
            emotions_dp = gr.Dropdown(EMOTION_NAMES, label="Choose emotion", value="Neutral")
            pitch_tb = gr.Textbox(label="Pitch values.")
            energy_tb = gr.Textbox(label="Energy values.")
            duration_tb = gr.Textbox(label="Pitch values.")
            dbfs_num = gr.Number(value=-27, label="Target dBFS")
            sample_speaker_cb = gr.Checkbox(label="Sample new speaker with GMM")
            use_last_cb = gr.Checkbox(label="Use the last sampled embedding")
            sample_noise_num = gr.Number(value=0.05, label="GMM noise scale")
            pitch_flatten_cb = gr.Checkbox(label="Pitch Flatten (Pitch * 0)")
            pitch_invert_cb = gr.Checkbox(label="Pitch Invert (Pitch * -1)")
            pitch_amplify_num = gr.Number(value=1.0, label="Pitch Amplify (Pitch * Pitch Amplify)")
            pitch_shift_num = gr.Number(value=0.0, label="Pitch Shift (Pitch + Pitch Shift in Hz)")
            speed_num = gr.Number(value=1.0, label="Speed")
            denoise_num = gr.Number(value=0.05, label="Denoise strength")
            denoise_method_dp = gr.Dropdown(["zeros", "normal"], label="Denoiser method", value="zeros")
            noise_scale_num = gr.Number(value=0.44, label="Model Noise Scale")
            split_batching_cb = gr.Checkbox(label="Split batching. (Split into sentences and batch)", value=True)
            tts_btn = gr.Button(label="Synthesize")

        with gr.Column() as col2:
             # Output Components
            output_audio = gr.Audio(label="Output Speech.")
            original_audio = gr.Audio(label="Original Speech.")
            spec_plot = gr.Plot(label="Spectrogram")
            orgspec_plot = gr.Plot(label="Org. Spectrogram")
            uploaded_plot = gr.Audio(label="Uploaded Speech.")
            durations_plot = gr.Plot(label="Durations")
            gt_pitch_plot = gr.Plot(label="GT pitch")
            pred_pitch_plot = gr.Plot(label="Predicted pitch")
            pitch_avg_plot = gr.Plot(label="Pitch Avg GT")

    tts_btn.click(
        fn=tts,
        inputs=[
            mic_audio,
            upload_file,
            sentence_tb,
            speakers_dp,
            emotions_dp,
            dbfs_num,
            sample_speaker_cb,
            use_last_cb,
            sample_noise_num,
            pitch_flatten_cb,
            pitch_invert_cb,
            pitch_amplify_num,
            pitch_shift_num,
            pitch_tb,
            energy_tb,
            speed_num,
            duration_tb,
            denoise_num,
            denoise_method_dp,
            noise_scale_num,
            split_batching_cb,
            state_vars,
        ],
        outputs=[
            output_audio,
            original_audio,
            pitch_tb,
            energy_tb,
            duration_tb,
            spec_plot,
            orgspec_plot,
            uploaded_plot,
            durations_plot,
            gt_pitch_plot,
            pred_pitch_plot,
            pitch_avg_plot,
            state_vars
        ],
    )

    sentence_tb.change(fn=clear_controls, inputs=None, outputs=[pitch_tb, energy_tb, duration_tb])

if __name__ == "__main__":
    demo.launch(
        share=False,
        debug=True,
        server_port=port,
        server_name="0.0.0.0",
        enable_queue=False,
    )
