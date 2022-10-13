import math
import os
import numpy as np
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from librosa import pyin
import parselmouth
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.trainer_utils import get_optimizer, get_scheduler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import F0Dataset, TTSDataset, _parse_sample
from TTS.tts.layers.generic.duration_predictor_lstm import BottleneckLayerLayer, DurationPredictorLSTM
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import ContextEncoder, PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import (
    average_over_durations,
    generate_path,
    maximum_path,
    rand_segments,
    segment,
    sequence_mask,
    compute_attn_prior,
)
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.utils.visual import plot_alignment, plot_avg_pitch, plot_pitch, plot_spectrogram
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results
from TTS.utils.audio.numpy_transforms import compute_f0

##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path, sample_rate=None):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(
        file_path,
    )
    assert (x > 1).sum() + (x < -1).sum() == 0
    if sample_rate:
        x = torchaudio.functional.resample(
            x,
            orig_freq=sr,
            new_freq=sample_rate,
        )
        sr = sample_rate
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def _wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    y = y.squeeze(1)

    # if torch.min(y) < -1.0:
    #     print("min value is ", torch.min(y))
    # if torch.max(y) > 1.0:
    #     print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    return spec


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    with torch.cuda.amp.autocast(enabled=False):
        spec = _wav_to_spec(y, n_fft, hop_length, win_length, center=center)

        if spec.dtype in [torch.cfloat, torch.cdouble]:
            spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    return spec


def wav_to_energy(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    mel = wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=center)
    return mel_to_energy(mel)

def mel_to_energy(mel):
    avg_energy = torch.mean(mel, dim=1, keepdim=True)
    avg_energy = (avg_energy + 20.0) / 20.0
    return avg_energy

def name_mel_basis(spec, n_fft, fmax):
    n_fft_len = f"{n_fft}_{fmax}_{spec.dtype}_{spec.device}"
    return n_fft_len


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    mel_basis_key = name_mel_basis(spec, n_fft, fmax)
    if mel_basis_key not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[mel_basis_key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[mel_basis_key], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T_y]`

    Return Shapes:
        - spec : :math:`[B,C,T_spec]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    mel_basis_key = name_mel_basis(y, n_fft, fmax)
    wnsize_dtype_device = str(win_length) + "_" + str(y.dtype) + "_" + str(y.device)
    if mel_basis_key not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[mel_basis_key] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[mel_basis_key], spec)
    spec = amp_to_db(spec)
    return spec


#############################
# CONFIGS
#############################
@dataclass
class VitsAudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None
    pitch_fmax: float = 640.0
    pitch_fmin: float = 80.0


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create inverse frequency weights for balancing the dataset.
    Use `multi_dict` to scale relative weights."""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        # check if all keys are in the multi_dict
        for k in multi_dict:
            assert k in unique_attr_names, f"{k} not in {unique_attr_names}"
        # scale weights
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class VitsF0Dataset(F0Dataset):
    """Override F0Dataset to avoid the AudioProcessor."""

    def __init__(
        self,
        audio_config: "AudioConfig",
        samples: Union[List[List], List[Dict]],
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        sample_rate=None,
    ):
        self.sample_rate = sample_rate
        super().__init__(
            samples=samples,
            audio_config=audio_config,
            verbose=verbose,
            cache_path=cache_path,
            precompute_num_workers=precompute_num_workers,
            normalize_f0=False,
        )

    @staticmethod
    def _compute_and_save_pitch(audio_config, wav_file, pitch_file=None, sample_rate=None):
        wav, current_sample_rate = load_audio(wav_file, sample_rate=sample_rate)
        # compute f0 using librosa
        f0, voiced_mask, _ = pyin(
            wav.numpy()[0], audio_config.pitch_fmin, audio_config.pitch_fmax, current_sample_rate,
            frame_length=audio_config.win_length * 2, win_length=audio_config.win_length,
            hop_length=audio_config.hop_length)
        f0[~voiced_mask] = 0.0
        # skip the last F0 value to align with the spectrogram
        if wav.shape[1] % audio_config.hop_length != 0:
            f0 = f0[:-1]
        if pitch_file:
            np.save(pitch_file, f0)

        # snd = parselmouth.Sound(wav_file)
        # # resample if needed
        # if sample_rate:
        #     snd = snd.resample(sample_rate)
        # # compute pitch
        # f0 = snd.to_pitch().selected_array['frequency']

        # # interpolate to match the spectrogram shape
        # spec_size = int(snd.values.shape[-1] / audio_config.hop_length)
        # f0 = torch.nn.functional.interpolate(torch.tensor(f0).unsqueeze(0).unsqueeze(0), scale_factor=(spec_size/len(f0),)).squeeze().numpy()
        # if pitch_file:
        #     np.save(pitch_file, f0)
        return f0

    def compute_or_load(self, wav_file):
        """
        compute pitch and return a numpy array of pitch values
        """
        pitch_file = self.create_pitch_file_path(wav_file, self.cache_path)
        if not os.path.exists(pitch_file):
            pitch = self._compute_and_save_pitch(
                audio_config=self.audio_config, wav_file=wav_file, pitch_file=pitch_file, sample_rate=self.sample_rate
            )
        else:
            pitch = np.load(pitch_file)
        return pitch.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        f0 = self.compute_or_load(item["audio_file"])
        return {"audio_file": item["audio_file"], "f0": f0}


class VitsDataset(TTSDataset):
    def __init__(self, *args, **kwargs):
        compute_f0 = kwargs.pop("compute_f0", False)
        self.encoder_sample_rate = kwargs.pop("encoder_sample_rate", False)
        kwargs["compute_f0"] = False
        self.attn_prior_cache_path = kwargs.pop("attn_prior_cache_path")
        super().__init__(*args, **kwargs)
        self.compute_f0 = compute_f0
        self.pad_id = self.tokenizer.characters.pad_id
        self.audio_config = kwargs["audio_config"]
        self.upsample_factor = self.audio_config.sample_rate // self.encoder_sample_rate

        if self.compute_f0:
            self.f0_dataset = VitsF0Dataset(
                audio_config=self.audio_config,
                samples=self.samples,
                cache_path=kwargs["f0_cache_path"],
                precompute_num_workers=kwargs["precompute_num_workers"],
                sample_rate=self.encoder_sample_rate,
            )

        if self.attn_prior_cache_path is not None:
            os.makedirs(self.attn_prior_cache_path, exist_ok=True)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # prevent unexpected matches by keeping all the folders in the file name
        rel_wav_path = Path(item["audio_file"]).relative_to(item["root_path"]).with_suffix("")
        rel_wav_path = str(rel_wav_path).replace("/", "_")

        raw_text = item["text"]
        wav, _ = load_audio(item["audio_file"])
        wav_filename = os.path.basename(item["audio_file"])

        if self.encoder_sample_rate is not None:
            remain = (wav.size(1) // self.audio_config.hop_length) % self.upsample_factor
            if remain > 0:
                wav = wav[:, : -int(self.audio_config.hop_length) * remain]

        token_ids = self.get_token_ids(idx, item["text"])

        f0 = None
        if self.compute_f0:
            f0 = self.get_f0(idx)["f0"]

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        # compute attn prior
        attn_prior = None
        if self.attn_prior_cache_path is not None:
            attn_prior = self.load_or_compute_attn_prior(token_ids, wav, rel_wav_path)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "pitch": f0,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
            "attn_prior": attn_prior,
        }

    def load_or_compute_attn_prior(self, token_ids, wav, rel_wav_path):
        """Load or compute and save the attention prior."""
        attn_prior_file = os.path.join(self.attn_prior_cache_path, f"{rel_wav_path}.npy")
        if os.path.exists(attn_prior_file):
            return np.load(attn_prior_file)
        else:
            token_len = len(token_ids)
            mel_len = wav.shape[1] // (self.audio_config.hop_length * self.upsample_factor)
            attn_prior = compute_attn_prior(token_len, mel_len)
            np.save(attn_prior_file, attn_prior)
            return attn_prior

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - pitch :math:`[B, T]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
            - attn_prior: :math:`[[T_token, T_mel]]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        pitch_padded = None
        if self.compute_f0:
            pitch_lens = [p.shape[0] for p in batch["pitch"]]
            pitch_lens = torch.LongTensor(pitch_lens)
            pitch_lens_max = torch.max(pitch_lens)
            pitch_padded = torch.FloatTensor(B, 1, pitch_lens_max)
            pitch_padded = pitch_padded.zero_() + self.pad_id

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)

        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id

        for i in range(B):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

            if self.compute_f0:
                pitch = batch["pitch"][i]
                pitch_padded[i, 0, : len(pitch)] = torch.FloatTensor(pitch)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "pitch": pitch_padded,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
            "attn_priors": batch["attn_prior"] if batch["attn_prior"][0] is not None else None,
        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class VitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels of the decoder. Defaults to 513.

        spec_segment_size (int):
            Decoder input segment size. Defaults to 32 `(32 * hoplength = waveform length)`.

        hidden_channels (int):
            Number of hidden channels of the model. Defaults to 192.

        hidden_channels_ffn_text_encoder (int):
            Number of hidden channels of the feed-forward layers of the text encoder transformer. Defaults to 256.

        num_heads_text_encoder (int):
            Number of attention heads of the text encoder transformer. Defaults to 2.

        num_layers_text_encoder (int):
            Number of transformer layers in the text encoder. Defaults to 6.

        kernel_size_text_encoder (int):
            Kernel size of the text encoder transformer FFN layers. Defaults to 3.

        dropout_p_text_encoder (float):
            Dropout rate of the text encoder. Defaults to 0.1.

        dropout_p_duration_predictor (float):
            Dropout rate of the duration predictor. Defaults to 0.1.

        kernel_size_posterior_encoder (int):
            Kernel size of the posterior encoder's WaveNet layers. Defaults to 5.

        dilatation_posterior_encoder (int):
            Dilation rate of the posterior encoder's WaveNet layers. Defaults to 1.

        num_layers_posterior_encoder (int):
            Number of posterior encoder's WaveNet layers. Defaults to 16.

        kernel_size_flow (int):
            Kernel size of the Residual Coupling layers of the flow network. Defaults to 5.

        dilatation_flow (int):
            Dilation rate of the Residual Coupling WaveNet layers of the flow network. Defaults to 1.

        num_layers_flow (int):
            Number of Residual Coupling WaveNet layers of the flow network. Defaults to 6.

        resblock_type_decoder (str):
            Type of the residual block in the decoder network. Defaults to "1".

        resblock_kernel_sizes_decoder (List[int]):
            Kernel sizes of the residual blocks in the decoder network. Defaults to `[3, 7, 11]`.

        resblock_dilation_sizes_decoder (List[List[int]]):
            Dilation sizes of the residual blocks in the decoder network. Defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.

        upsample_rates_decoder (List[int]):
            Upsampling rates for each concecutive upsampling layer in the decoder network. The multiply of these
            values must be equal to the kop length used for computing spectrograms. Defaults to `[8, 8, 2, 2]`.

        upsample_initial_channel_decoder (int):
            Number of hidden channels of the first upsampling convolution layer of the decoder network. Defaults to 512.

        upsample_kernel_sizes_decoder (List[int]):
            Kernel sizes for each upsampling layer of the decoder network. Defaults to `[16, 16, 4, 4]`.

        periods_multi_period_discriminator (List[int]):
            Periods values for Vits Multi-Period Discriminator. Defaults to `[2, 3, 5, 7, 11]`.

        use_sdp (bool):
            Use Stochastic Duration Predictor. Defaults to True.

        noise_scale (float):
            Noise scale used for the sample noise tensor in training. Defaults to 1.0.

        inference_noise_scale (float):
            Noise scale used for the sample noise tensor in inference. Defaults to 0.667.

        length_scale (float):
            Scale factor for the predicted duration values. Smaller values result faster speech. Defaults to 1.

        noise_scale_dp (float):
            Noise scale used by the Stochastic Duration Predictor sample noise in training. Defaults to 1.0.

        inference_noise_scale_dp (float):
            Noise scale for the Stochastic Duration Predictor in inference. Defaults to 0.8.

        max_inference_len (int):
            Maximum inference length to limit the memory use. Defaults to None.

        init_discriminator (bool):
            Initialize the disciminator network if set True. Set False for inference. Defaults to True.

        use_spectral_norm_disriminator (bool):
            Use spectral normalization over weight norm in the discriminator. Defaults to False.

        use_speaker_embedding (bool):
            Enable/Disable speaker embedding for multi-speaker models. Defaults to False.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

        detach_dp_input (bool):
            Detach duration predictor's input from the network for stopping the gradients. Defaults to True.

        use_language_embedding (bool):
            Enable/Disable language embedding for multilingual models. Defaults to False.

        embedded_language_dim (int):
            Number of language embedding channels. Defaults to 4.

        num_languages (int):
            Number of languages for the language embedding layer. Defaults to 0.

        language_ids_file (str):
            Path to the language mapping file for the Language Manager. Defaults to None.

        use_speaker_encoder_as_loss (bool):
            Enable/Disable Speaker Consistency Loss (SCL). Defaults to False.

        speaker_encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        speaker_encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

        condition_dp_on_speaker (bool):
            Condition the duration predictor on the speaker embedding. Defaults to True.

        freeze_encoder (bool):
            Freeze the encoder weigths during training. Defaults to False.

        freeze_DP (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        freeze_PE (bool):
            Freeze the posterior encoder weigths during training. Defaults to False.

        freeze_flow_encoder (bool):
            Freeze the flow encoder weigths during training. Defaults to False.

        freeze_waveform_decoder (bool):
            Freeze the waveform decoder weigths during training. Defaults to False.

        encoder_sample_rate (int):
            If not None this sample rate will be used for training the Posterior Encoder,
            flow, text_encoder and duration predictor. The decoder part (vocoder) will be
            trained with the `config.audio.sample_rate`. Defaults to None.

        interpolate_z (bool):
            If `encoder_sample_rate` not None and  this parameter True the nearest interpolation
            will be used to upsampling the latent variable z with the sampling rate `encoder_sample_rate`
            to the `config.audio.sample_rate`. If it is False you will need to add extra
            `upsample_rates_decoder` to match the shape. Defaults to True.

    """

    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    emotion_vector_file: str = None
    use_emotion_vector_file: bool = False
    emotion_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False
    encoder_sample_rate: int = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False
    use_pitch: bool = False
    pitch_predictor_hidden_channels: int = 192
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    use_energy_predictor: bool = False
    energy_predictor_hidden_channels: int = 256
    energy_predictor_kernel_size: int = 3
    energy_predictor_dropout_p: float = 0.1
    energy_embedding_kernel_size: int = 3
    use_context_encoder: bool = False


# @torch.jit.script
# def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
#     n_channels_int = n_channels[0]
#     in_act = input_a + input_b
#     t_act = torch.tanh(in_act[:, :n_channels_int, :])
#     s_act = torch.sigmoid(in_act[:, n_channels_int:, :])x
#     acts = t_act * s_act
#     return acts


# class WN(torch.nn.Module):
#     """Wavenet layers with weight norm and no input conditioning.

#          |-----------------------------------------------------------------------------|
#          |                                    |-> tanh    -|                           |
#     res -|- conv1d(dilation) -> dropout -> + -|            * -> conv1d1x1 -> split -|- + -> res
#     g -------------------------------------|  |-> sigmoid -|                        |
#     o ----------------------------------------------------------------------------- + --------- o

#     Args:
#         in_channels (int): number of input channels.
#         hidden_channes (int): number of hidden channels.
#         kernel_size (int): filter kernel size for the first conv layer.
#         dilation_rate (int): dilations rate to increase dilation per layer.
#             If it is 2, dilations are 1, 2, 4, 8 for the next 4 layers.
#         num_layers (int): number of wavenet layers.
#         c_in_channels (int): number of channels of conditioning input.
#         dropout_p (float): dropout rate.
#         weight_norm (bool): enable/disable weight norm for convolution layers.
#     """

#     def __init__(
#         self,
#         in_channels,
#         hidden_channels,
#         kernel_size,
#         dilation_rate,
#         num_layers,
#         c_in_channels=0,
#         dropout_p=0,
#         weight_norm=True,
#     ):
#         super().__init__()
#         assert kernel_size % 2 == 1
#         assert hidden_channels % 2 == 0
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.dilation_rate = dilation_rate
#         self.num_layers = num_layers
#         self.c_in_channels = c_in_channels
#         self.dropout_p = dropout_p

#         self.in_layers = torch.nn.ModuleList()
#         self.res_skip_layers = torch.nn.ModuleList()
#         self.dropout = nn.Dropout(dropout_p)
#         # prenet for projecting inputs
#         pre_layer = torch.nn.Conv1d(in_channels+c_in_channels, hidden_channels, 1)
#         self.pre_layer  = torch.nn.utils.weight_norm(pre_layer, name='weight')
#         # init conditioning layer
#         if c_in_channels > 0:
#             cond_layer = torch.nn.Conv1d(c_in_channels, 2 * hidden_channels * num_layers, 1)
#             self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
#         # intermediate layers
#         for i in range(num_layers):
#             dilation = dilation_rate**i
#             padding = int((kernel_size * dilation - dilation) / 2)
#             in_layer = torch.nn.Conv1d(
#                 hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding
#             )
#             in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
#             self.in_layers.append(in_layer)

#             if i < num_layers - 1:
#                 res_skip_channels = 2 * hidden_channels
#             else:
#                 res_skip_channels = hidden_channels

#             res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
#             res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
#             self.res_skip_layers.append(res_skip_layer)
#         # setup weight norm
#         if not weight_norm:
#             self.remove_weight_norm()

#     def forward(self, x, x_mask=None, g=None, **kwargs):  # pylint: disable=unused-argument
#         x = torch.cat((x, g), 1)  # append context to z as well
#         x = self.start(x)
#         output = torch.zeros_like(x)
#         n_channels_tensor = torch.IntTensor([self.hidden_channels])
#         x_mask = 1.0 if x_mask is None else x_mask
#         if g is not None:
#             g = self.cond_layer(g)
#         for i in range(self.num_layers):
#             x_in = self.in_layers[i](x)
#             x_in = self.dropout(x_in)
#             if g is not None:
#                 cond_offset = i * 2 * self.hidden_channels
#                 g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
#             else:
#                 g_l = torch.zeros_like(x_in)
#             acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
#             res_skip_acts = self.res_skip_layers[i](acts)
#             if i < self.num_layers - 1:
#                 x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
#                 output = output + res_skip_acts[:, self.hidden_channels :, :]
#             else:
#                 output = output + res_skip_acts
#         return output * x_mask

#     def remove_weight_norm(self):
#         if self.c_in_channels != 0:
#             torch.nn.utils.remove_weight_norm(self.cond_layer)
#         for l in self.in_layers:
#             torch.nn.utils.remove_weight_norm(l)
#         for l in self.res_skip_layers:
#             torch.nn.utils.remove_weight_norm(l)


class Vits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
        emotion_manager: "EmotionManager" = None,
    ):

        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        self.emotion_manager = emotion_manager
        self.init_multispeaker(config)
        self.init_multilingual(config)
        self.init_emotion()
        self.init_upsampling()

        self.target_cps = 16
        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size
        self.binarize_alignment = False

        # embedding layer used only with the alignment network.
        # self.emb_aligner = nn.Embedding(self.args.num_chars, self.args.hidden_channels)
        # nn.init.normal_(self.emb_aligner.weight, 0.0, self.args.hidden_channels**-0.5)

        # --> TEXT ENCODER
        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
            emo_emb_dim=self.embedded_emotion_dim,
            padding_idx=self.tokenizer.characters.pad_id,
        )

        self.text_encoder_proj = nn.Conv1d(self.args.hidden_channels, self.args.hidden_channels * 2, 1)

        # --> ALIGNMENT NETWORK
        in_key_channels = self.args.hidden_channels
        if self.embedded_speaker_dim > 0 :
            self.aligner_spk_bottleneck = BottleneckLayerLayer(in_channels=self.embedded_speaker_dim, bottleneck_channels=32, norm="weight_norm")
            in_key_channels += 32
        if self.embedded_emotion_dim > 0 :
            in_key_channels += self.embedded_emotion_dim
        self.aligner = AlignmentNetwork(
            in_query_channels=self.args.out_channels, in_key_channels=in_key_channels
        )
        self.binary_loss_weight = 0.0

        # --> PITCH PREDICTOR
        context_cond_channels = 0
        if self.args.use_pitch:
            self.pitch_predictor = DurationPredictorLSTM(
                in_channels=self.args.hidden_channels,
                bottleneck_channels=16,  # TODO: make this configurable
                spk_emb_channels=self.embedded_speaker_dim,
                emo_emb_channels=self.embedded_emotion_dim,
            )
            self.pitch_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.pitch_embedding_kernel_size,
                padding=int((self.args.pitch_embedding_kernel_size - 1) / 2),
            )
            context_cond_channels += self.args.hidden_channels
            self.pitch_scaler = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=True, momentum=None)
            self.pitch_scaler.requires_grad_(False)

        # --> ENERGY PREDICTOR
        if self.args.use_energy_predictor:
            self.energy_predictor = DurationPredictorLSTM(
                in_channels=self.args.hidden_channels,
                bottleneck_channels=16,
                spk_emb_channels=self.embedded_speaker_dim,
                emo_emb_channels=self.embedded_emotion_dim,
            )
            self.energy_emb = nn.Conv1d(
                1,
                128,
                kernel_size=self.args.energy_embedding_kernel_size,
                padding=int((self.args.energy_embedding_kernel_size - 1) / 2),
            )
            # self.energy_scaler = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=True, momentum=None)
            # self.energy_scaler.requires_grad_(False)
            context_cond_channels += 128

        # --> CONTEXT ENCODER
        if self.args.use_context_encoder:
            self.context_encoder = ContextEncoder(
                in_channels=self.args.hidden_channels,
                cond_channels=context_cond_channels,
                spk_emb_channels=self.embedded_speaker_dim,
                emo_emb_channels=self.embedded_emotion_dim,
                num_lstm_layers=1,
                lstm_norm="spectral",
            )

        # --> FLOW STEPS
        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.context_encoder.hidden_lstm_channels * 2,
        )

        # --> POSTERIOR ENCODER
        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.context_encoder.hidden_lstm_channels * 2,
        )

        # --> DURATION PREDICTOR
        if self.args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if self.args.condition_dp_on_speaker else 0,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictorLSTM(
                self.args.hidden_channels,
                bottleneck_channels=48,
                spk_emb_channels=self.embedded_speaker_dim,
                emo_emb_channels=self.embedded_emotion_dim
            )

        # --> VOCODER
        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            # cond_channels=self.context_encoder.hidden_lstm_channels * 2,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio.sample_rate,
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def init_emotion(self):
        # pylint: disable=attribute-defined-outside-init
        self.embedded_emotion_dim = 0
        self.num_emotions = 0

        if self.emotion_manager:
            self.num_emotions = self.emotion_manager.num_emotions

        # if self.args.use_emotion_embedding:
        #     if self.num_emotions > 0:
        #         print(" > initialization of emotion-embedding layers.")
        #         self.emb_emotion = nn.Embedding(self.num_emotions, self.args.emotion_embedding_dim)
        #         self.embedded_emotion_dim += self.args.emotion_embedding_dim

        if self.args.use_emotion_vector_file:
            self.embedded_emotion_dim += self.args.emotion_vector_dim

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()
        # set the device of speaker encoder
        if self.args.use_speaker_encoder_as_loss:
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            print(" > Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            print(" > Text Encoder was reinit.")

    def on_epoch_end(self, trainer):
        # if self.args.use_energy_predictor:
        #     # stop updating mean and var
        #     self.energy_scaler.eval()
        if self.args.use_pitch:
            # stop updating mean and var
            self.pitch_scaler.eval()

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid, _ = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, emo_emb = None, None, None, None

        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)

        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "emotion_vectors" in aux_input and aux_input["emotion_vectors"] is not None:
            emo_emb = F.normalize(aux_input["emotion_vectors"]).unsqueeze(-1)
            if emo_emb.ndim == 2:
                emo_emb = emo_emb.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        return sid, g, lid, emo_emb

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_duration_predictor(self, outputs, attn, x, x_mask, g, lang_emb, emo_emb, x_lengths):
        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            # log_durations = self.duration_predictor(
            #     x.detach() if self.args.detach_dp_input else x,
            #     x_mask,
            #     g=g.detach() if self.args.detach_dp_input and g is not None else g,
            #     lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            # )
            log_durations = self.duration_predictor(
                txt_enc=x.detach() if self.args.detach_dp_input else x,
                spk_emb=g.detach() if self.args.detach_dp_input and g is not None else g,
                emo_emb=emo_emb.detach() if self.args.detach_dp_input and emo_emb is not None else emo_emb,
                lens=x_lengths,
            )
            # compute duration loss
            attn_log_durations = torch.log(attn_durations + 1) * x_mask
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
            outputs["log_duration_pred"] = log_durations.squeeze(1)  # [B, 1] -> [B]
        outputs["loss_duration"] = loss_duration
        return outputs

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        g: torch.FloatTensor = None,
        emo_emb: torch.FloatTensor = None,
        pitch_transform: Callable = None,
        x_lengths: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.
        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.
        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.
            g (torch.FloatTensor, optional): Conditioning vectors. In general speaker embeddings. Defaults to None.
            pitch_transform (Callable, optional): Pitch transform function. Defaults to None.
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.
        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
        # o_pitch = self.pitch_predictor(o_en, x_mask, g=g)
        o_pitch = self.pitch_predictor(
            txt_enc=o_en,
            spk_emb=g.detach(),
            emo_emb=emo_emb.detach(),
            lens=x_lengths,
        )

        if pitch is not None:
            # Training
            o_pitch_emb = self.pitch_emb(pitch)
            loss_pitch = torch.nn.functional.mse_loss(o_pitch * x_mask, pitch * x_mask, reduction="sum")
            loss_pitch = loss_pitch / x_mask.sum()
            return o_pitch_emb, o_pitch, loss_pitch
        else:
            # Inference
            if pitch_transform is not None:
                o_pitch = pitch_transform(o_pitch, x_mask.sum(dim=(1, 2)), self.pitch_mean, self.pitch_std)
            o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch

    def _forward_energy_predictor(
        self,
        o_en: torch.FloatTensor,

        x_mask: torch.IntTensor,
        energy: torch.FloatTensor = None,
        g: torch.FloatTensor = None,
        emo_emb: torch.FloatTensor = None,
        energy_transform: Callable = None,
        x_lengths: torch.IntTensor = None,
    ) -> torch.FloatTensor:
        o_energy = self.energy_predictor(
            txt_enc=o_en,
            spk_emb=g.detach(),
            emo_emb=emo_emb.detach(),
            lens=x_lengths,
        )
        if energy is not None:
            # Training
            gt_avg_energy = energy
            gt_avg_energy = gt_avg_energy * 2 - 1  # scale to ~ [-1, 1]
            gt_avg_energy = gt_avg_energy * 1.4  # scale to ~ 1 std
            o_energy_emb = self.energy_emb(energy)
            loss_energy = torch.nn.functional.mse_loss(o_energy * x_mask, gt_avg_energy * x_mask, reduction="sum")
            loss_energy = loss_energy / x_mask.sum()
            return o_energy_emb, o_energy, loss_energy
        else:
            # Inference
            # denormalize predicted energy
            o_energy = o_energy / 1.4
            o_energy = (o_energy + 1) / 2
            if energy_transform is not None:
                o_energy = energy_transform(o_energy, x_mask.sum(dim=(1, 2)), self.pitch_mean, self.pitch_std)
            o_energy_emb = self.energy_emb(o_energy)
            return o_energy_emb, o_energy

    def _forward_aligner(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        x_mask: torch.IntTensor,
        y_mask: torch.IntTensor,
        attn_priors: torch.FloatTensor,
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Aligner forward pass.
        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.
        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.
            attn_priors (torch.FloatTensor): Prior for the aligner network map.
        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.
        Shapes:
            - x: :math:`[B, C_en, T_en]`
            - y: :math:`[B, C_de, T_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`
            - attn_priors: :math:`[B, T_de, T_en]`
            - aligner_durations: :math:`[B, T_en]`
            - aligner_soft: :math:`[B, T_de, T_en]`
            - aligner_logprob: :math:`[B, 1, T_de, T_en]`
            - aligner_mas: :math:`[B, T_de, T_en]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)  # [B, 1, T_max, T_max2]
        aligner_soft, aligner_logprob = self.aligner(queries=y, keys=x, mask=x_mask, attn_prior=attn_priors)
        aligner_hard = maximum_path(
            aligner_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        aligner_durations = torch.sum(aligner_hard, -1)
        aligner_soft = aligner_soft.transpose(2, 3) * attn_mask  # [B, 1, T_max2, T_max] -> [B, 1, T_max, T_max2]
        aligner_hard = aligner_hard[:, None]  # [B, T_max, T_max2] -> [B, 1, T_max, T_max2]
        return aligner_durations, aligner_soft, aligner_logprob, aligner_hard

    def _expand_encoder(
        self, o_en: torch.FloatTensor, y_lengths: torch.IntTensor, dr: torch.IntTensor, x_mask: torch.FloatTensor
    ):
        """Expand encoder outputs to match the decoder.

        1. Compute the decoder output mask
        2. Expand encoder output with the durations.
        3. Apply position encoding.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            y_lengths (torch.IntTensor): Output sequence lengths.
            dr (torch.IntTensor): Ground truth durations or alignment network durations.
            x_mask (torch.IntTensor): Input sequence mask.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: Decoder mask, expanded encoder outputs,
                attention map
        """
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(
            en=o_en, dr=dr, x_mask=x_mask, y_mask=y_mask
        )  # [B, T_de', C_en], [B, T_en, T_de']
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
        return y_mask, o_en_ex, attn.transpose(1, 2)

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        pitch: torch.tensor,
        energy: torch.tensor,
        waveform: torch.tensor,
        attn_priors: torch.tensor,
        binarize_alignment: bool,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        """Forward pass of the model.

        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of linear spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - emotion_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        outputs = {}
        sid, spk_emb, lid, emo_emb = self._set_cond_input(aux_input)

        if self.config.add_blank:
            blank_mask = x == self.tokenizer.characters.blank_id
            blank_mask[:, 0] = False

        ##### --> EMBEDDINGS

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            spk_emb = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        ##### --> TEXT ENCODING

        x_emb, o_en, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb, emo_emb=emo_emb)

        ##### --> ALIGNER
        # x_emb_aligner = self.emb_aligner(x) * math.sqrt(self.args.hidden_channels)
        # x_emb_aligner = x_emb_aligner.transpose(1, 2)  # [B, T, C] --> [B, C, T]
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).float()
        spk_emb_bottle = self.aligner_spk_bottleneck(spk_emb)  # [B, C, 1]
        spk_emb_bottle_ex = spk_emb_bottle.expand(-1, -1, x_emb.shape[2])
        emo_emb_ex = emo_emb.expand(-1, -1, x_emb.shape[2])
        x_emb_spk = torch.cat((x_emb, spk_emb_bottle_ex.detach()), 1)
        x_emb_spk = torch.cat((x_emb_spk, emo_emb_ex.detach()), 1)
        aligner_durations, aligner_soft, aligner_logprob, aligner_hard = self._forward_aligner(
            x=x_emb_spk, y=y, x_mask=x_mask, y_mask=y_mask, attn_priors=attn_priors
        )

        # duration predictor and duration loss
        outputs = self.forward_duration_predictor(outputs, aligner_hard, o_en, x_mask, spk_emb, lang_emb, emo_emb, x_lengths)

        ##### --> EXPAND

        # expand prior
        if binarize_alignment:
            m_p = torch.einsum("klmn, kjm -> kjn", [aligner_hard, m_p])
            logs_p = torch.einsum("klmn, kjm -> kjn", [aligner_hard, logs_p])
            o_en_ex = torch.einsum("klmn, kjm -> kjn", [aligner_hard, o_en])
        else:
            m_p = torch.einsum("klmn, kjm -> kjn", [aligner_soft, m_p])
            logs_p = torch.einsum("klmn, kjm -> kjn", [aligner_soft, logs_p])
            o_en_ex = torch.einsum("klmn, kjm -> kjn", [aligner_soft, o_en])

        ##### --> Pitch & Energy Predictors

        # pitch predictor pass
        o_pitch = None
        avg_pitch = None
        if self.args.use_pitch:
            # remove extra frame if needed
            if energy.size(2) != pitch.size(2):
                pitch = pitch[:, :, :energy.size(2)]
            o_pitch_emb, o_pitch, loss_pitch = self._forward_pitch_predictor(
                o_en=o_en_ex, x_mask=y_mask, pitch=pitch, g=spk_emb, emo_emb=emo_emb, x_lengths=y_lengths
            )


        # energy predictor pass
        o_energy = None
        avg_energy = None
        if self.args.use_energy_predictor:
            o_energy_emb, o_energy, loss_energy = self._forward_energy_predictor(
                o_en=o_en_ex, x_mask=y_mask, energy=energy, g=spk_emb, emo_emb=emo_emb, x_lengths=y_lengths
            )

        ##### ---> CONTEXT ENCODING

        # context encoder pass
        context_cond = torch.cat((o_energy_emb, o_pitch_emb), dim=1)  # [B, C * 2, T_de]
        context_emb = self.context_encoder(o_en_ex, y_lengths, spk_emb=spk_emb, emo_emb=emo_emb, cond=context_cond)

        ###### --> FLOW

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=context_emb * y_mask)

        # flow layers
        z_p = self.flow(z, y_mask, g=context_emb * y_mask)

        ###### --> VOCODER

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)
        # context_emb_seg = segment(
        #     context_emb * y_mask,
        #     slice_ids,
        #     spec_segment_size,
        #     pad_short=True,
        # )

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)
        # context_emb_seg, _, _, _ = self.upsampling_z(context_emb_seg, slice_ids=slice_ids)

        # TODO: Try passing only spk_emb
        o = self.waveform_decoder(z_slice, g=None)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None
        outputs.update(
            {
                "model_outputs": o,
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "energy_hat": o_energy,
                "pitch_hat": o_pitch,
                "loss_pitch": loss_pitch,
                "loss_energy": loss_energy,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
                "aligner_soft": aligner_soft.squeeze(1),
                "aligner_hard": aligner_hard.squeeze(1),
                "aligner_durations": aligner_durations,
                "aligner_logprob": aligner_logprob,
            }
        )
        return outputs

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.no_grad()
    def inference(
        self,
        x,
        pitch_transform=None,
        energy_transform=None,
        aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None},
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        sid, spk_emb, lid, emo_emb = self._set_cond_input(aux_input)
        x_lengths = self._set_x_lengths(x, aux_input)

        if self.config.add_blank:
            blank_mask = x == self.tokenizer.characters.blank_id
            blank_mask[:, 0] = False

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            spk_emb = self.emb_g(sid).unsqueeze(-1)

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        _, o_en, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb, emo_emb=emo_emb)

        if self.args.use_sdp:
            dur_log = self.duration_predictor(
                o_en,
                x_mask,
                g=spk_emb if self.args.condition_dp_on_speaker else None,
                reverse=True,
                noise_scale=self.inference_noise_scale_dp,
                lang_emb=lang_emb,
            )
        else:
            # dur_log = self.duration_predictor(
            #     o_en, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
            # )
            dur_log = self.duration_predictor(o_en, spk_emb=spk_emb, emo_emb=emo_emb, lens=x_lengths)

        dur = (torch.exp(dur_log) - 1) * x_mask * self.length_scale

        if self.target_cps:
            model_output_in_sec = (self.config.audio.hop_length * dur.sum()) / self.args.encoder_sample_rate
            num_input_chars = x_lengths[0]
            num_input_chars = num_input_chars - (x == self.tokenizer.characters.char_to_id(" ")).sum()
            num_input_chars -= blank_mask.sum()
            dur = dur / (self.target_cps / (num_input_chars / model_output_in_sec))
            model_output_in_sec2 = (self.config.audio.hop_length * dur.sum()) / self.args.encoder_sample_rate

        dur = torch.round(dur)
        dur[dur < 1] = 1

        y_lengths = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(dur.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        o_en_ex = torch.matmul(attn.transpose(1, 2), o_en.transpose(1, 2)).transpose(1, 2)

        # pitch predictor pass
        o_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch = self._forward_pitch_predictor(
                o_en=o_en_ex,
                x_mask=y_mask,
                g=spk_emb,
                emo_emb=emo_emb,
                pitch_transform=pitch_transform,
                x_lengths=y_lengths,
            )

        # energy predictor pass
        o_energy = None
        if self.args.use_energy_predictor:
            o_energy_emb, o_energy = self._forward_energy_predictor(
                o_en=o_en_ex,
                x_mask=y_mask,
                g=spk_emb,
                emo_emb=emo_emb,
                energy_transform=energy_transform,
                x_lengths=y_lengths,
            )

        # context encoder pass
        context_cond = torch.cat((o_energy_emb, o_pitch_emb), dim=1)  # [B, C * 2, T_en]
        o_context = self.context_encoder(o_en_ex, y_lengths, spk_emb=spk_emb, emo_emb=emo_emb, cond=context_cond)

        # if self.args.use_pitch:
        #     o_en = o_en + o_pitch_emb
        # if self.args.use_energy_predictor:
        #     o_en = o_en + o_energy_emb

        # stats = self.text_encoder_proj(o_en) * x_mask
        # m_p, logs_p = torch.split(stats, self.args.hidden_channels, dim=1)

        # if self.args.use_sdp:
        #     dur_log = self.duration_predictor(
        #         o_en,
        #         x_mask,
        #         g=spk_emb if self.args.condition_dp_on_speaker else None,
        #         reverse=True,
        #         noise_scale=self.inference_noise_scale_dp,
        #         lang_emb=lang_emb,
        #     )
        # else:
        #     dur_log = self.duration_predictor(
        #         o_en, x_mask, g=spk_emb if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
        #     )

        # dur = (torch.exp(dur_log) -1) * x_mask * self.length_scale
        # dur[dur < 1] = 1.0
        # dur = torch.round(dur)


        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=o_context * y_mask, reverse=True)

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)
        # o_context, _, _, _ = self.upsampling_z(o_context, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=None)

        masked_dur = dur.int()
        masked_dur[:, :, 1:-1] = masked_dur[:, :, 1:-1] + masked_dur[:, :, 2:]
        masked_dur = masked_dur.masked_fill(blank_mask, 0.0)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": dur,
            "masked_durations": masked_dur,
            "pitch": o_pitch,
            "energy": o_energy,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
        }
        return outputs

    @torch.no_grad()
    def inference_voice_conversion(
        self, reference_wav, speaker_id=None, d_vector=None, reference_speaker_id=None, reference_d_vector=None
    ):
        """Inference for voice conversion

        Args:
            reference_wav (Tensor): Reference wavform. Tensor of shape [B, T]
            speaker_id (Tensor): speaker_id of the target speaker. Tensor of shape [B]
            d_vector (Tensor): d_vector embedding of target speaker. Tensor of shape `[B, C]`
            reference_speaker_id (Tensor): speaker_id of the reference_wav speaker. Tensor of shape [B]
            reference_d_vector (Tensor): d_vector embedding of the reference_wav speaker. Tensor of shape `[B, C]`
        """
        # compute spectrograms
        y = wav_to_spec(
            reference_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        )
        y_lengths = torch.tensor([y.size(-1)]).to(y.device)
        speaker_cond_src = reference_speaker_id if reference_speaker_id is not None else reference_d_vector
        speaker_cond_tgt = speaker_id if speaker_id is not None else d_vector
        wav, _, _ = self.voice_conversion(y, y_lengths, speaker_cond_src, speaker_cond_tgt)
        return wav

    def voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        TODO: create an end-point for voice conversion

        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Reference speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(speaker_cond_src).unsqueeze(-1)
            g_tgt = self.emb_g(speaker_cond_tgt).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """

        spec_lens = batch["spec_lens"]

        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            spec = batch["spec"]
            energy = batch["energy"]
            pitch = batch["pitch"]
            d_vectors = batch["d_vectors"]
            emotion_vectors = batch["emotion_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            waveform = batch["waveform"]
            attn_priors = batch["attn_priors"]

            # generator pass
            outputs = self.forward(
                x=tokens,
                x_lengths=token_lenghts,
                y=spec,
                y_lengths=spec_lens,
                pitch=pitch,
                energy=energy,
                waveform=waveform,
                attn_priors=attn_priors,
                binarize_alignment=self.binarize_alignment,
                aux_input={
                    "d_vectors": d_vectors,
                    "speaker_ids": speaker_ids,
                    "language_ids": language_ids,
                    "emotion_vectors": emotion_vectors,
                },
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):

                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    token_len=batch["token_lens"],
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    log_duration_pred=self.model_outputs_cache["log_duration_pred"].float(),
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    aligner_logprob=self.model_outputs_cache["aligner_logprob"].float(),
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                    alignment_hard=self.model_outputs_cache["aligner_hard"],
                    alignment_soft=self.model_outputs_cache["aligner_soft"],
                    binary_loss_weight=self.binary_loss_weight,
                )

            loss_dict["loss"] = self.model_outputs_cache["loss_pitch"] * self.config.pitch_loss_alpha + loss_dict["loss"]
            loss_dict["loss"] = self.model_outputs_cache["loss_energy"] * self.config.energy_loss_alpha + loss_dict["loss"]
            loss_dict["loss_pitch"] = self.model_outputs_cache["loss_pitch"] * self.config.pitch_loss_alpha
            loss_dict["loss_energy"] = self.model_outputs_cache["loss_energy"] * self.config.energy_loss_alpha

            loss_dict["avg_text_length"] = batch["token_lens"].float().mean()
            loss_dict["avg_spec_length"] = batch["spec_lens"].float().mean()
            loss_dict["avg_text_batch_occupancy"] = (
                batch["token_lens"].float() / batch["token_lens"].float().max()
            ).mean()
            loss_dict["avg_spec_batch_occupancy"] = (
                batch["spec_lens"].float() / batch["spec_lens"].float().max()
            ).mean()

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/audio": sample_voice}

        # # plot pitch figures
        # if self.args.use_pitch:
        #     pitch_avg = abs(outputs[1]["pitch_avg_gt"][0, 0].data.cpu().numpy())
        #     pitch_avg_hat = abs(outputs[1]["pitch_avg"][0, 0].data.cpu().numpy())
        #     chars = self.tokenizer.decode(batch["tokens"][0].data.cpu().numpy())
        #     pitch_figures = {
        #         "pitch_ground_truth": plot_avg_pitch(pitch_avg, chars, output_fig=False),
        #         "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, chars, output_fig=False),
        #     }
        #     figures.update(pitch_figures)

        # # plot energy figures
        # if self.args.use_energy_predictor:
        #     energy_avg = abs(outputs[1]["energy_avg_gt"][0, 0].data.cpu().numpy())
        #     energy_avg_hat = abs(outputs[1]["energy_avg"][0, 0].data.cpu().numpy())
        #     chars = self.tokenizer.decode(batch["tokens"][0].data.cpu().numpy())
        #     energy_figures = {
        #         "energy_ground_truth": plot_avg_pitch(energy_avg, chars, output_fig=False),
        #         "energy_avg_predicted": plot_avg_pitch(energy_avg_hat, chars, output_fig=False),
        #     }
        #     figures.update(energy_figures)

        attn_hard = outputs[1]["aligner_hard"]
        attn_hard = attn_hard[0].data.cpu().numpy().T
        attn_soft = outputs[1]["aligner_soft"]
        attn_soft = attn_soft[0].data.cpu().numpy().T
        figures.update(
            {
                "alignment_hard": plot_alignment(attn_hard, output_fig=False),
                "alignment_soft": plot_alignment(attn_soft, output_fig=False),
            }
        )
        return figures, audios

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        figures, audios = self._log(self.ap, batch, outputs, "train")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._log(self.ap, batch, outputs, "eval")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
            elif len(sentence_info) == 5:
                text, speaker_name, style_wav, language_name, emotion = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get emotion id/vector
        emotion_vector = None
        if hasattr(self, "speaker_manager"):
            if config.use_emotion_vector_file:
                if emotion is None:
                    emotion_vector = self.emotion_manager.get_random_embeddings()
                else:
                    emotion_vector = self.emotion_manager.get_mean_embedding(emotion, num_samples=None, randomize=False)

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
            "emotion_vector": emotion_vector,
        }

    def plot_outputs(self, text, wav, alignment, outputs):
        figures = {}
        # plot pitch and spectrogram
        if self.args.encoder_sample_rate:
            try:
                wav = torchaudio.functional.resample(torch.from_numpy(wav[None, :]),
                    orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
                ).squeeze(1)
            except:
                wav = torchaudio.functional.resample(torch.from_numpy(wav[None, :]).cuda(),
                    orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
                ).cpu().squeeze(1)
        else:
            wav = torch.from_numpy(wav[None, :])

        spec = wav_to_mel(
            y=wav,
            n_fft=self.config.audio.fft_size,
            sample_rate=self.config.audio.sample_rate,
            num_mels=self.config.audio.num_mels,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax,
            center=False,
        )[0].transpose(0, 1)

        y = torch.nn.functional.pad(
            wav.unsqueeze(1),
            (int((self.config.audio.fft_size - self.config.audio.hop_length) / 2), int((self.config.audio.fft_size - self.config.audio.hop_length) / 2)),
            mode="reflect",
        )
        y = y.squeeze().numpy()

        pitch, voiced_mask, _ = pyin(
            y, self.config.audio.pitch_fmin, self.config.audio.pitch_fmax, self.config.audio.sample_rate if not self.args.encoder_sample_rate else self.args.encoder_sample_rate,
            frame_length=self.config.audio.win_length * 2, win_length=self.config.audio.win_length,
            hop_length=self.config.audio.hop_length)

        pitch[~voiced_mask] = 0.0

        input_text = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(text, language="en"))
        input_text = input_text.replace("<BLNK>", "_")

        durations = outputs["durations"]

        pitch_avg = average_over_durations(
            torch.from_numpy(pitch)[None, None, :], durations.squeeze(0).cpu()
        )  # [1, 1, n_frames]

        pred_pitch = outputs["pitch"].squeeze().cpu()

        figures["alignment"] = plot_alignment(alignment.transpose(1, 2), output_fig=False)
        figures["spectrogram"] = plot_spectrogram(spec)
        figures["pitch_from_audio"] = plot_pitch(pitch, spec)
        figures["pitch_predicted"] = plot_pitch(pred_pitch, spec)
        figures["pitch_avg_from_audio"] = plot_avg_pitch(pitch_avg.squeeze(), input_text)
        return figures

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            return_dict = self.synthesize(
                aux_inputs["text"],
                speaker_id=aux_inputs["speaker_id"],
                language_id=aux_inputs["language_id"],
                emotion_id=None,
                d_vector=aux_inputs["d_vector"],
                ref_waveform=aux_inputs["style_wav"],
                emotion_vector=aux_inputs["emotion_vector"],
            )

            # plot outputs
            wav = return_dict["wav"]
            alignment = return_dict["alignments"]
            test_audios["{}-audio".format(idx)] = wav.T

            # plot pitch and spectrogram
            if self.args.encoder_sample_rate:
                try:
                    wav = self.audio_resampler(torch.from_numpy(wav[None, :])).squeeze(1)
                except:
                    wav = self.audio_resampler(torch.from_numpy(wav[None, :]).cuda()).cpu().squeeze(1)
            else:
                wav = torch.from_numpy(wav[None, :])

            spec = wav_to_mel(
                y=wav,
                n_fft=self.config.audio.fft_size,
                sample_rate=self.config.audio.sample_rate,
                num_mels=self.config.audio.num_mels,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
                center=False,
            )[0].transpose(0, 1)

            pitch, voiced_mask, _ = pyin(
                wav.numpy()[0], self.config.audio.pitch_fmin, self.config.audio.pitch_fmax, self.config.audio.sample_rate if not self.args.encoder_sample_rate else self.args.encoder_sample_rate,
                frame_length=self.config.audio.win_length * 2, win_length=self.config.audio.win_length,
                hop_length=self.config.audio.hop_length)
            pitch[~voiced_mask] = 0.0

            input_text = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(aux_inputs["text"], language="en"))
            input_text = input_text.replace("<BLNK>", "_")

            durations = return_dict["outputs"]["durations"]

            pitch_avg = average_over_durations(
                torch.from_numpy(pitch)[None, None, :], durations.squeeze(0).cpu()
            )  # [1, 1, n_frames]

            pred_pitch = return_dict["outputs"]["pitch"].squeeze().cpu()

            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.transpose(1, 2), output_fig=False)
            test_figures["{}-spectrogram".format(idx)] = plot_spectrogram(spec)
            test_figures["{}-pitch_from_audio".format(idx)] = plot_pitch(pitch, spec)
            test_figures["{}-pitch_predicted".format(idx)] = plot_pitch(pred_pitch, spec)
            test_figures["{}-pitch_avg_from_audio".format(idx)] = plot_avg_pitch(pitch_avg.squeeze(), input_text)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None
        emo_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.name_to_id and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.name_to_id[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get emotion_vectors from audio file names
        if self.emotion_manager is not None and self.emotion_manager.embeddings and self.args.use_emotion_vector_file:
            vector_mapping = self.emotion_manager.embeddings
            emo_vectors = [vector_mapping[w]["embedding"] for w in batch["audio_files"]]
            emo_vectors = torch.FloatTensor(emo_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["emotion_vectors"] = emo_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with target sampling rate for the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)

        # attn priors
        if self.config.use_attn_priors:
            attn_priors_np = batch["attn_priors"]

            batch["attn_priors"] = torch.zeros(
                batch["spec"].shape[0],
                batch["spec_lens"].max(),
                batch["token_lens"].max(),
                device=batch["mel"].device,
            )

            for i in range(batch["mel"].shape[0]):
                batch["attn_priors"][i, : attn_priors_np[i].shape[0], : attn_priors_np[i].shape[1]] = torch.from_numpy(
                    attn_priors_np[i]
                )

        # compute energy
        batch["energy"] = None
        if self.args.use_energy_predictor:
            energy_mel_spec = wav_to_mel(
                wav,
                n_fft=ac.fft_size,
                num_mels=ac.num_mels,
                sample_rate=ac.sample_rate if not self.args.encoder_sample_rate else self.args.encoder_sample_rate,
                hop_length=ac.hop_length,
                win_length=ac.win_length,
                fmin=ac.mel_fmin,
                fmax=ac.mel_fmax,
                center=False
            )
            batch["energy"] = mel_to_energy(energy_mel_spec)
            # batch["energy"] = self.energy_scaler(batch["energy"])

        if self.args.use_pitch:
            zero_idxs = batch["pitch"] == 0.0
            pitch_norm = self.pitch_scaler(batch["pitch"])
            pitch_norm[zero_idxs] = 0.0
            batch["pitch"] = pitch_norm[: ,: , :batch["energy"].shape[-1]]
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
<<<<<<< HEAD
<<<<<<< HEAD
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
=======
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items
>>>>>>> Use self.synthesize in test_run
=======
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
>>>>>>> Add attribute weigtening
                )
                weights = weights * alpha
                print(f" > Attribute weights for '{attr_names}' \n | > {attr_weights}")

        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None
        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = VitsDataset(
                # model_args=self.args,
                samples=samples,
                audio_config=self.config.audio,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                compute_f0=config.compute_f0,
                f0_cache_path=config.f0_cache_path,
                attn_prior_cache_path=config.attn_prior_cache_path if config.use_attn_priors else None,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                encoder_sample_rate=config.model_args.encoder_sample_rate,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)
            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,  # shuffle is done in the dataset.
                    collate_fn=dataset.collate_fn,
                    drop_last=False,  # setting this False might cause issues in AMP training.
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}

        if self.args.encoder_sample_rate is not None and eval:
            # audio resampler is not used in inference time
            self.audio_resampler = None

        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            print(f" > Loading checkpoint with {num_new_speakers} additional speakers.")
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            self.set_inference()
            assert not self.training

    def set_inference(self):
        self.pitch_mean = self.pitch_scaler.running_mean.clone()
        self.pitch_std = torch.sqrt(self.pitch_scaler.running_var.clone())
        # self.energy_mean = self.energy_scaler.running_mean.clone()
        # self.energy_std = torch.sqrt(self.energy_scaler.running_var.clone())

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)
        emotion_manager = EmotionManager.init_from_config(config.model_args)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
            )
        return Vits(new_config, ap, tokenizer, speaker_manager, language_manager, emotion_manager)

    def synthesize(
        self,
        text: str,
        speaker_id,
        language_id,
        d_vector,
        ref_waveform,
        emotion_vector=None,
        emotion_id=None,
        pitch_transform=None,
        noise_scale=0.66,
    ):
        # TODO: add language_id
        is_cuda = next(self.parameters()).is_cuda

        # convert text to sequence of token IDs
        text_len = len(text)
        text_inputs = np.asarray(
            self.tokenizer.text_to_ids(text, language=language_id),
            dtype=np.int32,
        )

        # pass tensors to backend
        if speaker_id is not None:
            speaker_id = id_to_torch(speaker_id, cuda=is_cuda)

        if d_vector is not None:
            d_vector = embedding_to_torch(d_vector, cuda=is_cuda)

        if emotion_vector is not None:
            emotion_vector = embedding_to_torch(emotion_vector, cuda=is_cuda)

        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=is_cuda)
        text_inputs = text_inputs.unsqueeze(0)

        # synthesize voice
        self.inference_noise_scale = noise_scale
        outputs = self.inference(
            text_inputs,
            aux_input={"d_vectors": d_vector, "speaker_ids": speaker_id, "emotion_vectors": emotion_vector},
            pitch_transform=pitch_transform,
        )

        # collect outputs
        wav = outputs["model_outputs"][0].data.cpu().numpy()
        alignments = outputs["alignments"]
        return_dict = {
            "wav": wav,
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }
        return return_dict


def id_to_torch(aux_id, cuda=False):
    if aux_id is not None:
        aux_id = np.asarray(aux_id)
        aux_id = torch.from_numpy(aux_id)
    if cuda:
        return aux_id.cuda()
    return aux_id


def embedding_to_torch(d_vector, cuda=False):
    if d_vector is not None:
        d_vector = np.asarray(d_vector)
        d_vector = torch.from_numpy(d_vector).type(torch.FloatTensor)
        d_vector = d_vector.squeeze().unsqueeze(0)
    if cuda:
        return d_vector.cuda()
    return d_vector


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor


##################################
# VITS CHARACTERS
##################################


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(graphemes, punctuations, pad, None, None, "<BLNK>", is_unique=False, is_sorted=True)

    def _create_vocab(self):
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    @staticmethod
    def init_from_config(config: Coqpit):
        if config.characters is not None:
            _pad = config.characters["pad"]
            _punctuations = config.characters["punctuations"]
            _letters = config.characters["characters"]
            _letters_ipa = config.characters["phonemes"]
            return (
                VitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = VitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=None,
            bos=None,
            blank=self._blank,
            is_unique=False,
            is_sorted=True,
        )


# if __name__ == "__main__":
#     from TTS.tts.configs.vits_config import VitsConfig

#     def _create_inputs(config, batch_size=2, device="cuda"):
#         input_dummy = torch.randint(0, 24, (batch_size, 10)).long().to(device)
#         input_lengths = torch.randint(2, 11, (batch_size,)).long().to(device)
#         input_lengths[-1] = 10
#         spec = torch.rand(batch_size, config.audio["fft_size"] // 2 + 1, 30).to(device)
#         mel = torch.rand(batch_size, config.audio["num_mels"], 30).to(device)
#         spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
#         spec_lengths[-1] = spec.size(2)
#         waveform = torch.rand(batch_size, 1, spec.size(2) * config.audio["hop_length"]).to(device)
#         energy = torch.rand(batch_size, 1, 30).to(device)
#         pitch = torch.rand(batch_size, 1, 30).to(device)
#         spk_emb = torch.rand(batch_size, 512).to(device)
#         return input_dummy, input_lengths, mel, spec, spec_lengths, waveform, energy, pitch, spk_emb

#     config = VitsConfig()
#     config.model_args.use_pitch = True
#     config.model_args.use_energy_predictor = True
#     config.model_args.use_context_encoder = True
#     config.model_args.use_d_vector_file = True
#     config.model_args.d_vector_dim = 512
#     config.model_args.use_sdp = False
#     model = Vits.init_from_config(config)
#     model.cuda()

#     input_dummy, input_lengths, mel, spec, spec_lengths, waveform, energy, pitch, spk_emb = _create_inputs(config)
#     model.forward(
#         x=input_dummy,
#         x_lengths=input_lengths,
#         y=spec,
#         y_lengths=spec_lengths,
#         waveform=waveform,
#         energy=energy,
#         pitch=pitch,
#         attn_priors=None,
#         aux_input={"d_vectors": spk_emb},
#     )

#     model.inference(x=input_dummy[0][None, :], aux_input={"d_vectors": spk_emb[0][None, :]})
