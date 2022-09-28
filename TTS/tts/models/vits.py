import math
import os
import pysbd
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa import pyin
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils import remove_weight_norm
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.utils.samplers import BucketBatchSampler
from TTS.utils.io import load_fsspec
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import F0Dataset, TTSDataset, _parse_sample
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.layers.generic.duration_predictor_lstm import BottleneckLayerLayer, DurationPredictorLSTM
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import ContextEncoder, PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.layers.vits.denoise import VitsDenoiser
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.utils.helpers import (
    average_over_durations,
    compute_attn_prior,
    generate_path,
    maximum_path,
    rand_segments,
    segment,
    sequence_mask,
)
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_pitch, plot_pitch, plot_spectrogram
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

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
    weight_normed = False
    if hasattr(m, "weight_g"):
        weight_normed = True
        torch.nn.utils.remove_weight_norm(m)
    if callable(reset_parameters):
        print(" > Reseting weights of {}".format(m))
        m.reset_parameters()
    if weight_normed:
        torch.nn.utils.weight_norm(m)


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
        # add the padding for compute the pitch
        y = torch.nn.functional.pad(
            wav.unsqueeze(1),
            (
                int((audio_config.fft_size - audio_config.hop_length) / 2),
                int((audio_config.fft_size - audio_config.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze().numpy()
        f0, voiced_mask, _ = pyin(
            y.astype(np.double),
            audio_config.pitch_fmin,
            audio_config.pitch_fmax,
            current_sample_rate,
            frame_length=audio_config.win_length,
            win_length=audio_config.win_length // 2,
            hop_length=audio_config.hop_length,
            pad_mode="reflect",
            center=False,
            n_thresholds=100,
            beta_parameters=(2, 18),
            boltzmann_parameter=2,
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
        )
        f0[~voiced_mask] = 0.0

        if pitch_file:
            np.save(pitch_file, f0)
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
            "wav_path": item["audio_file"],
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
            "audio_paths": batch["wav_path"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
            "attn_priors": batch["attn_prior"] if batch["attn_prior"][0] is not None else None,
        }


##############################
# MODEL DEFINITION
##############################


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def stride_lens(lens: torch.Tensor, stride: int = 2) -> torch.Tensor:
    return torch.ceil(lens / stride).int()


class StyleEmbedAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, num_units: int, num_heads: int):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query: torch.Tensor, key_soft: torch.Tensor) -> torch.Tensor:
        """
        input:
            query --- [N, T_q, query_dim]
            key_soft --- [N, T_k, key_dim]
        output:
            out --- [N, T_q, num_units]
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        out_soft = scores_soft = None
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key_soft)  # [N, T_k, num_units]

        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores_soft = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores_soft = scores_soft / (self.key_dim**0.5)
        scores_soft = F.softmax(scores_soft, dim=3)

        # out = score * V
        # [h, N, T_q, num_units/h]
        out_soft = torch.matmul(scores_soft, values)
        out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out_soft  # , scores_soft


class STL(nn.Module):
    """Style Token Layer"""

    def __init__(self, hidden_channels: int, num_heads: int = 1, num_tokens: int = 32):
        super(STL, self).__init__()

        num_heads = 1
        self.embed = nn.Parameter(torch.FloatTensor(num_tokens, hidden_channels // num_heads))
        d_q = hidden_channels // 2
        d_k = hidden_channels // num_heads
        self.attention = StyleEmbedAttention(query_dim=d_q, key_dim=d_k, num_units=hidden_channels, num_heads=num_heads)

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        query = x.unsqueeze(1)  # [N, 1, E//2]

        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]

        # Weighted sum
        emotion_embed_soft = self.attention(query, keys_soft)

        return emotion_embed_soft


class AddCoords(nn.Module):
    """
    https://arxiv.org/pdf/1807.03247.pdf
    https://github.com/mkocabas/CoordConv-pytorch
    """

    def __init__(self, rank: int, with_r: bool = False):
        super().__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = x.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            xx_channel = xx_channel.to(x.device)
            out = torch.cat([x, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)

            out = torch.cat([x, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)
            zz_channel = zz_channel.to(x.device)
            out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2) + torch.pow(zz_channel - 0.5, 2)
                )
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(torch.nn.modules.conv.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        kernel_size: int = 3,
        hidden_channels: List[int] = [32, 32, 64, 64, 128, 128],
        strides: int = [1, 2, 1, 2, 1],
        gpu_channels: int = 32,
    ):
        super().__init__()

        K = len(hidden_channels)
        filters = [num_mels] + hidden_channels
        strides = [1] + strides
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [
            CoordConv1d(
                in_channels=filters[0],
                out_channels=filters[0 + 1],
                kernel_size=kernel_size,
                stride=strides[0],
                padding=kernel_size // 2,
                with_r=True,
            )
        ]
        convs2 = [
            nn.Conv1d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=kernel_size,
                stride=strides[i],
                padding=kernel_size // 2,
            )
            for i in range(1, K)
        ]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([nn.InstanceNorm1d(num_features=hidden_channels[i], affine=True) for i in range(K)])

        self.gru = nn.GRU(
            input_size=hidden_channels[-1],
            hidden_size=gpu_channels,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, mel_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs --- [N,  n_mels, timesteps]
        outputs --- [N, E//2]
        """

        mel_masks = get_mask_from_lengths(mel_lens).unsqueeze(1)
        x = x.masked_fill(mel_masks, 0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.leaky_relu(x, 0.3)  # [N, 128, Ty//2^K, n_mels//2^K]
            x = norm(x)

        for _ in range(2):
            mel_lens = stride_lens(mel_lens)

        mel_masks = get_mask_from_lengths(mel_lens)

        x = x.masked_fill(mel_masks.unsqueeze(1), 0)
        x = x.permute((0, 2, 1))
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mel_lens.cpu().int(), batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        x, memory = self.gru(x)  # memory --- [N, Ty, E//2], out --- [1, N, E//2]
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x, memory, mel_masks

    def calculate_channels(self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int) -> int:
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class UtteranceLevelProsodyEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 256,
        gru_channels: int = 32,
        p_dropout: float = 0.2,
        bottleneck_channels: int = 256,
        num_stl_heads: int = 1,
        num_stl_tokens: int = 32,
        reference_encoder_num_mels: int = 80,
        reference_encoder_hidden_channels: List[int] = [32, 32, 64, 64, 128, 128],
        reference_encoder_strides: int = [1, 2, 1, 2, 1],
        reference_encoder_kernel_size: int = 3,
        reference_encoder_gpu_channels: int = 32,
    ):
        super().__init__()
        self.encoder = ReferenceEncoder(
            num_mels=reference_encoder_num_mels,
            kernel_size=reference_encoder_kernel_size,
            hidden_channels=reference_encoder_hidden_channels,
            strides=reference_encoder_strides,
            gpu_channels=reference_encoder_gpu_channels,
        )
        self.encoder_prj = nn.Linear(gru_channels, hidden_channels // 2)
        self.stl = STL(hidden_channels=hidden_channels, num_heads=num_stl_heads, num_tokens=num_stl_tokens)
        self.encoder_bottleneck = nn.Linear(hidden_channels, bottleneck_channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        """
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, E]
        """
        _, embedded_prosody, _ = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Style Token
        out = self.encoder_bottleneck(self.stl(embedded_prosody))
        out = self.dropout(out)

        out = out.view((-1, 1, out.shape[3]))
        return out


class BSConv1d(nn.Module):
    """https://arxiv.org/pdf/2003.13549.pdf"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int):
        super().__init__()
        self.pointwise = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.pointwise(x)
        x2 = self.depthwise(x1)
        return x2


class ConvTransposed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = BSConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


class PhonemeProsodyPredictor(nn.Module):
    """Non-parallel Prosody Predictor inspired by Du et al., 2021"""
    # TODO: use LSTM duration predictor like architecture

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, p_dropout: float):
        super().__init__()
        kernel_size = kernel_size
        dropout = p_dropout
        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(0.3),
                nn.LayerNorm(hidden_channels),  # TODO: consider instance norm as in StyleTTS
                nn.Dropout(dropout),
                ConvTransposed(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(0.3),
                nn.LayerNorm(hidden_channels),  # TODO: consider instance norm as in StyleTTS
                nn.Dropout(dropout),
            ]
        )
        self.predictor_bottleneck = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x -- [B, src_len, d_model]
        mask -- [B, src_len, 1]
        outputs -- [B, src_len, 2 * d_model]
        """
        for layer in self.layers:
            x = layer(x)
        x = x.masked_fill(mask, 0.0)
        x = self.predictor_bottleneck(x)
        return x


def positional_encoding(d_model: int, length: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


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

        d_vector_file (List[str]):
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

        freeze_duration_predictor (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        feezer_pitch_predictor (bool):
            Freeze the pitch predictor weigths during training. Defaults to False.

        freeze_energy_predictor (bool):
            Freeze the energy predictor weigths during training. Defaults to False.

        freeze_posterior_encoder (bool):
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
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    combd_channels: List = field(default_factory=lambda: [16, 64, 256, 1024, 1024, 1024])
    combd_kernels: List = field(
        default_factory=lambda: [[7, 11, 11, 11, 11, 5], [11, 21, 21, 21, 21, 5], [15, 41, 41, 41, 41, 5]]
    )
    combd_groups: List = field(default_factory=lambda: [1, 4, 16, 64, 256, 1])
    combd_strides: List = field(default_factory=lambda: [1, 1, 4, 4, 4, 1])
    tkernels: List = field(default_factory=lambda: [7, 5, 3])
    fkernel: int = 5
    tchannels: List = field(default_factory=lambda: [64, 128, 256, 256, 256])
    fchannels: List = field(default_factory=lambda: [32, 64, 128, 128, 128])
    tstrides: List = field(default_factory=lambda: [[1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1]])
    fstride: List = field(default_factory=lambda: [1, 1, 3, 3, 1])
    tdilations: List = field(
        default_factory=lambda: [
            [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
            [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        ]
    )
    fdilations: List = field(default_factory=lambda: [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]])
    pqmf_n: int = 16
    pqmf_m: int = 64
    freq_init_ch: int = 256
    tsubband: List = field(default_factory=lambda: [6, 11, 16])
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: List[str] = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    emotion_vector_file: List[str] = None
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
    freeze_duration_predictor: bool = False
    freeze_pitch_predictor: bool = False
    freeze_energy_predictor: bool = False
    freeze_posterior_encoder: bool = False
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


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        u_bias = self.u_bias.expand_as(query)
        v_bias = self.v_bias.expand_as(query)
        a = (query + u_bias).transpose(1, 2)
        content_score = a @ key.transpose(2, 3)
        b = (query + v_bias).transpose(1, 2)
        pos_score = b @ pos_embedding.permute(0, 2, 3, 1)
        pos_score = self._relative_shift(pos_score)

        score = content_score + pos_score
        score = score * (1.0 / self.sqrt_dim)

        score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)

        context = (attn @ value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = torch.zeros((batch_size, num_heads, seq_length1, 1), device=pos_score.device)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score


class ConformerMultiHeadedSelfAttention(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.attention = RelativeMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = key.size()
        encoding = encoding[:, : key.shape[1]]
        encoding = encoding.repeat(batch_size, 1, 1)
        outputs, attn = self.attention(query, key, value, pos_embedding=encoding, mask=mask)
        outputs = self.dropout(outputs)
        return outputs, attn


class PhonemeLevelProsodyEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        bottleneck_channels: int = 16,
        gru_channels: int = 32,
        num_heads: int = 2,
        p_dropout: float = 0.1,
        reference_encoder_num_mels: int = 80,
        reference_encoder_hidden_channels: List[int] = [32, 32, 64, 64, 128, 128],
        reference_encoder_strides: int = [1, 2, 1, 2, 1],
        reference_encoder_kernel_size: int = 3,
        reference_encoder_gpu_channels: int = 32,
    ):
        super().__init__()

        self.E = hidden_channels
        self.d_q = self.d_k = hidden_channels
        bottleneck_size = bottleneck_channels
        ref_enc_gru_size = gru_channels

        self.encoder = ReferenceEncoder(
            num_mels=reference_encoder_num_mels,
            kernel_size=reference_encoder_kernel_size,
            hidden_channels=reference_encoder_hidden_channels,
            strides=reference_encoder_strides,
            gpu_channels=reference_encoder_gpu_channels,
        )
        self.encoder_prj = nn.Linear(ref_enc_gru_size, hidden_channels)
        self.attention = ConformerMultiHeadedSelfAttention(
            d_model=hidden_channels,
            num_heads=num_heads,
            dropout_p=p_dropout,
        )
        self.encoder_bottleneck = nn.Linear(hidden_channels, bottleneck_size)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        x --- [N, seq_len, encoder_embedding_dim]
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, bottleneck_size]
        attn --- [N, seq_len, ref_len], Ty/r = ref_len
        """
        embedded_prosody, _, mel_masks = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        attn_mask = mel_masks.view((mel_masks.shape[0], 1, 1, -1))
        x, _ = self.attention(
            query=x,
            key=embedded_prosody,
            value=embedded_prosody,
            mask=attn_mask,
            encoding=encoding,
        )
        x = self.encoder_bottleneck(x)
        x = x.masked_fill(src_mask, 0.0)
        return x


def concat_embeddings(context, speaker_embedding, emotion_embedding):
    spk_emb_bottle_ex = speaker_embedding.expand(-1, -1, context.shape[2])  # [B, C, T]
    emo_emb_ex = emotion_embedding.expand(-1, -1, context.shape[2])  # [B, C, T]
    return torch.cat((context, spk_emb_bottle_ex, emo_emb_ex), 1)  # [B, C, T]


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
            padding_idx=self.tokenizer.characters.pad_id,
        )

        self.text_encoder_proj = nn.Conv1d(self.args.hidden_channels, self.args.hidden_channels * 2, 1)

        # --> ALIGNMENT NETWORK
        in_key_channels = self.args.hidden_channels
        if self.embedded_speaker_dim > 0:
            self.aligner_spk_bottleneck = BottleneckLayerLayer(
                in_channels=self.embedded_speaker_dim, bottleneck_channels=32, norm="weight_norm"
            )
            in_key_channels += 32
        if self.embedded_emotion_dim > 0:
            in_key_channels += self.embedded_emotion_dim
        self.aligner = AlignmentNetwork(in_query_channels=self.args.out_channels, in_key_channels=in_key_channels)
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
            # self. = ResidualCouplingBlocks(
            #     self.args.hidden_channels,
            #     self.args.hidden_channels,
            #     kernel_size=self.args.kernel_size_flow,
            #     dilation_rate=self.args.dilation_rate_flow,
            #     num_layers=self.args.num_layers_flow,
            #     cond_channels=self.context_encoder.hidden_lstm_channels * 2,
            # )
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

        ## -- BOTTLENECK ADAPTOPRS -- ##

        in_channels_with_emb = self.args.hidden_channels
        if self.embedded_speaker_dim > 0:
            self.adaptors_spk_bottleneck = BottleneckLayerLayer(
                in_channels=self.embedded_speaker_dim, bottleneck_channels=32, norm="weight_norm"
            )
            in_channels_with_emb += 32
        if self.embedded_emotion_dim > 0:
            in_channels_with_emb += self.embedded_emotion_dim

        ## -- UTTRANCE ENCODER & PREDICTOR -- ##

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            hidden_channels=self.args.hidden_channels,
            gru_channels=32,
            p_dropout=0.2,
            bottleneck_channels=96,
            num_stl_heads=1,
            num_stl_tokens=32,
            reference_encoder_num_mels=513,
            reference_encoder_hidden_channels=[32, 32, 64, 64, 128, 128],
            reference_encoder_strides=[1, 2, 1, 2, 1],
            reference_encoder_kernel_size=3,
            reference_encoder_gpu_channels=32,
        )
        self.utterance_prosody_predictor = PhonemeProsodyPredictor(
            in_channels=in_channels_with_emb,
            hidden_channels=self.args.hidden_channels,
            out_channels=96,  # TODO: config these
            kernel_size=3,
            p_dropout=0.1,
        )
        self.u_bottle_out = nn.Linear(
            96,
            self.args.hidden_channels,
        )
        # TODO: change this with IntanceNorm as in StyleTTS
        self.u_norm = nn.LayerNorm(
            96,
            elementwise_affine=False,
        )

        ## -- PHONEME ENCODER & PREDICTOR -- ##

        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            self.args.hidden_channels,
            bottleneck_channels=16,
            gru_channels=32,
            num_heads=2,
            p_dropout=0.1,
            reference_encoder_num_mels=513,
            reference_encoder_hidden_channels=[32, 32, 64, 64, 128, 128],
            reference_encoder_strides=[1, 2, 1, 2, 1],
            reference_encoder_kernel_size=3,
            reference_encoder_gpu_channels=32,
        )
        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(
            in_channels=in_channels_with_emb,
            hidden_channels=self.args.hidden_channels,
            out_channels=16,
            kernel_size=5,
            p_dropout=0.1,
        )
        self.p_bottle_out = nn.Linear(
            16,
            self.args.hidden_channels,
        )
        # TODO: change this with IntanceNorm as in StyleTTS
        self.p_norm = nn.LayerNorm(
            16,
            elementwise_affine=False,
        )

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
                emo_emb_channels=self.embedded_emotion_dim,
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
            self.disc = VitsDiscriminator(self.args)

        # denoiser only to be used in inference
        self.denoiser = VitsDenoiser(self)


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

    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        if self.config.binary_alignment_loss_warmup_epochs > 0:
            self.binary_loss_weight = (
                min(trainer.epochs_done / self.config.binary_alignment_loss_warmup_epochs, 1.0) * 1.0
            )
        else:
            self.binary_loss_weight = 1.0
        self.binarize_alignment = trainer.total_steps_done > self.config.binarization_start_steps

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(f" [!] The weights in Duration Predictor - {key} was not reinit check it !")
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
        sid, g, lid = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_posterior_encoder:
            print(" > Freezing posterior encoder...")
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_duration_predictor:
            print(" > Freezing duration predictor...")
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_energy_predictor:
            print(" > Freezing energy predictor...")
            for param in self.energy_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_pitch_predictor:
            print(" > Freezing pitch predictor...")
            for param in self.pitch_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            print(" > Freezing flow decoder...")
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            print(" > Freezing waveform decoder...")
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

    def average_utterance_prosody(self, u_prosody_pred: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        mel: torch.tensor,
        pitch: torch.tensor,
        energy: torch.tensor,
        waveform: torch.tensor,
        attn_priors: torch.tensor,
        binarize_alignment: bool,
        use_encoder: bool = True,
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
        x_filler_mask = ~(x_mask.bool())

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
        outputs = self.forward_duration_predictor(
            outputs, aligner_hard, o_en, x_mask, spk_emb, lang_emb, emo_emb, x_lengths
        )

        ##### --> Bottleneck Embeddings

        spk_emb_adaptors = self.adaptors_spk_bottleneck(spk_emb)  # [B, C, 1]
        o_en1 = concat_embeddings(o_en, spk_emb_adaptors, emo_emb)  # [B, C, 1]

        ##### --> Utterance Prosody encoder
        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(mels=y, mel_lens=y_lengths)
        )  # TODO: use mel like the original imp
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(
                    x=o_en1.transpose(1, 2), mask=x_filler_mask.transpose(1, 2)
                ),
                lengths=x_lengths,
            )
        )

        if use_encoder:
            o_en = o_en + self.u_bottle_out(u_prosody_ref).transpose(1, 2)
        else:
            o_en = o_en + self.u_bottle_out(u_prosody_pred).transpose(1, 2)

        # compute loss
        u_prosody_loss = 0.5 * torch.nn.functional.mse_loss(u_prosody_ref.detach(), u_prosody_pred)

        ##### --> Phoneme prosody encoder

        o_en2 = concat_embeddings(o_en, spk_emb_adaptors, emo_emb)

        pos_encoding = positional_encoding(
            self.args.hidden_channels,
            max(x_emb.shape[2], max(y_lengths)),
            device=x_emb.device,
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=o_en.transpose(1, 2),
                src_mask=x_filler_mask.transpose(1, 2),
                mels=y,
                mel_lens=y_lengths,
                encoding=pos_encoding,
            )
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=o_en2.transpose(1, 2), mask=x_filler_mask.transpose(1, 2))
        )

        if use_encoder:
            o_en = o_en + self.p_bottle_out(p_prosody_ref).transpose(1, 2)
        else:
            o_en = o_en + self.p_bottle_out(p_prosody_pred).transpose(1, 2)

        # compute loss
        p_prosody_loss = 0.5 * torch.nn.functional.mse_loss(
            p_prosody_ref.masked_select(x_mask.bool().transpose(1, 2)).detach(),
            p_prosody_pred.masked_select(x_mask.bool().transpose(1, 2)),
        )

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
                pitch = pitch[:, :, : energy.size(2)]
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
        o, o1, o2 = self.waveform_decoder(z_slice, g=None)

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
                "o1": o1,
                "o2": o2,
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
                "p_prosody_ref": p_prosody_ref,
                "u_prosody_ref": u_prosody_ref,
                "u_prosody_loss": u_prosody_loss,
                "p_prosody_loss": p_prosody_loss,
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
        x_lengths=None,
        pitch_transform=None,
        energy_transform=None,
        aux_input={"emotion_vectors": None, "d_vectors": None, "speaker_ids": None, "language_ids": None},
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - emotion_vectors: :math:`[B, C]`
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

        if x_lengths is None:
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

        ### --> TEXT ENCODER

        _, o_en, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb, emo_emb=emo_emb)
        x_filler_mask = ~(x_mask.bool())

        ### --> DURATION PREDICTOR

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
            dur_log = self.duration_predictor(o_en, spk_emb=spk_emb, emo_emb=emo_emb, lens=x_lengths)

        ### --> DURATION FORMATTING
        dur = (torch.exp(dur_log) - 1) * x_mask
        if self.target_cps:
            target_sr = (
                self.args.encoder_sample_rate if self.args.encoder_sample_rate else self.args.encoder_sample_rate
            )
            model_output_in_sec = (self.config.audio.hop_length * dur.squeeze(1).sum(axis=1)) / target_sr
            num_input_chars = x_lengths.clone().float()
            num_input_chars = num_input_chars - (x == self.tokenizer.characters.char_to_id(" ")).sum(1).float()
            dur = dur / (self.target_cps / (num_input_chars / model_output_in_sec)[:, None, None])

        # check hack to fix fast jumps
        # dur[dur < dur.mean()] = dur.mean()
        dur = dur * self.length_scale
        dur = torch.round(dur)
        dur[dur < 1] = 1
        dur =  dur * x_mask


        y_lengths = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        ##### --> Bottleneck Embeddings

        spk_emb_adaptors = self.adaptors_spk_bottleneck(spk_emb)  # [B, C, 1]
        o_en1 = concat_embeddings(o_en, spk_emb_adaptors, emo_emb)  # [B, C, 1]

        ##### --> Utterance Prosody encoder
        # u_prosody_ref = self.u_norm(self.utterance_prosody_encoder(mels=y, mel_lens=y_lengths))  # TODO: use mel like the original imp
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(
                    x=o_en1.transpose(1, 2), mask=x_filler_mask.transpose(1, 2)
                ),
                lengths=x_lengths,
            )
        )

        o_en = o_en + self.u_bottle_out(u_prosody_pred).transpose(1, 2)

        ##### --> Phoneme prosody encoder

        o_en2 = concat_embeddings(o_en, spk_emb_adaptors, emo_emb)

        pos_encoding = positional_encoding(
            self.args.hidden_channels,
            max(o_en.shape[2], max(y_lengths)),
            device=o_en.device,
        )

        # p_prosody_ref = self.p_norm(
        #     self.phoneme_prosody_encoder(
        #         x=o_en.transpose(1, 2), src_mask=x_filler_mask.transpose(1, 2), mels=y, mel_lens=y_lengths, encoding=pos_encoding
        #     )
        # )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=o_en2.transpose(1, 2), mask=x_filler_mask.transpose(1, 2))
        )

        o_en = o_en + self.p_bottle_out(p_prosody_pred).transpose(1, 2)

        ### --> ATTENTION MAP
        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(dur.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        ### --> EXPAND

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        o_en_ex = torch.matmul(attn.transpose(1, 2), o_en.transpose(1, 2)).transpose(1, 2)

        ### --> PITCH & ENERGY PREDICTORS

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

        ### --> CONTEXT ENCODER

        # context encoder pass
        context_cond = torch.cat((o_energy_emb, o_pitch_emb), dim=1)  # [B, C * 2, T_en]
        o_context = self.context_encoder(o_en_ex, y_lengths, spk_emb=spk_emb, emo_emb=emo_emb, cond=context_cond)

        ### --> FLOW DECODER

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=o_context * y_mask, reverse=True)

        ### --> VOCODER

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)
        # o_context, _, _, _ = self.upsampling_z(o_context, y_lengths=y_lengths, y_mask=y_mask)

        o, _, _ = self.waveform_decoder((z * y_mask), g=None)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": dur,
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
                mel=batch["mel"],
                aux_input={
                    "d_vectors": d_vectors,
                    "speaker_ids": speaker_ids,
                    "language_ids": language_ids,
                    "emotion_vectors": emotion_vectors,
                },
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores
            scores_disc_real, scores_disc_fake, _, _ = self.disc(
                x=outputs["waveform_seg"],
                x_hat=outputs["model_outputs"].detach(),
                x1_hat=outputs["o1"].detach(),
                x2_hat=outputs["o2"].detach(),
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
            _, scores_disc_fake, feats_disc_real, feats_disc_fake = self.disc(
                x_hat=self.model_outputs_cache["model_outputs"],
                x=self.model_outputs_cache["waveform_seg"],
                x1_hat=self.model_outputs_cache["o1"],
                x2_hat=self.model_outputs_cache["o2"],
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

            # add pitch and energy losses
            loss_dict["loss"] = (
                self.model_outputs_cache["loss_pitch"] * self.config.pitch_loss_alpha + loss_dict["loss"]
            )
            loss_dict["loss"] = (
                self.model_outputs_cache["loss_energy"] * self.config.energy_loss_alpha + loss_dict["loss"]
            )
            loss_dict["loss_pitch"] = self.model_outputs_cache["loss_pitch"] * self.config.pitch_loss_alpha
            loss_dict["loss_energy"] = self.model_outputs_cache["loss_energy"] * self.config.energy_loss_alpha

            # add prosody losses
            loss_dict["loss"] = (
                self.model_outputs_cache["u_prosody_loss"] * self.config.u_prosody_loss_alpha + loss_dict["loss"]
            )
            loss_dict["loss"] = (
                self.model_outputs_cache["p_prosody_loss"] * self.config.p_prosody_loss_alpha + loss_dict["loss"]
            )
            loss_dict["loss_u_prosody"] = self.model_outputs_cache["u_prosody_loss"] * self.config.u_prosody_loss_alpha
            loss_dict["loss_p_prosody"] = self.model_outputs_cache["p_prosody_loss"] * self.config.p_prosody_loss_alpha

            # compute useful training stats
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
                    d_vector = self.speaker_manager.get_random_embeddings()
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
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav[None, :]),
                    orig_freq=self.config.audio["sample_rate"],
                    new_freq=self.args.encoder_sample_rate,
                ).squeeze(1)
            except:
                wav = (
                    torchaudio.functional.resample(
                        torch.from_numpy(wav[None, :]).cuda(),
                        orig_freq=self.config.audio["sample_rate"],
                        new_freq=self.args.encoder_sample_rate,
                    )
                    .cpu()
                    .squeeze(1)
                )
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
            (
                int((self.config.audio.fft_size - self.config.audio.hop_length) / 2),
                int((self.config.audio.fft_size - self.config.audio.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze().numpy()

        pitch, voiced_mask, _ = pyin(
            y.astype(np.double),
            self.config.audio.pitch_fmin,
            self.config.audio.pitch_fmax,
            self.config.audio.sample_rate if not self.args.encoder_sample_rate else self.args.encoder_sample_rate,
            frame_length=self.config.audio.win_length,
            win_length=self.config.audio.win_length // 2,
            hop_length=self.config.audio.hop_length,
            pad_mode="reflect",
            center=False,
            n_thresholds=100,
            beta_parameters=(2, 18),
            boltzmann_parameter=2,
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
        )

        pitch[~voiced_mask] = 0.0

        input_text = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(text, language="en"))
        input_text = input_text.replace("<BLNK>", "_")

        durations = outputs["durations"]

        pitch_avg = average_over_durations(
            torch.from_numpy(pitch)[None, None, :], durations[0].cpu()
        )  # [1, 1, n_frames]

        pred_pitch = outputs["pitch"][0].cpu()

        figures["alignment"] = plot_alignment(alignment.transpose(1, 2)[0], output_fig=False)
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

            y = torch.nn.functional.pad(
                wav.unsqueeze(1),
                (
                    int((self.config.audio.fft_size - self.config.audio.hop_length) / 2),
                    int((self.config.audio.fft_size - self.config.audio.hop_length) / 2),
                ),
                mode="reflect",
            )
            y = y.squeeze().numpy()

            pitch, voiced_mask, _ = pyin(
                y.astype(np.double),
                self.config.audio.pitch_fmin,
                self.config.audio.pitch_fmax,
                self.config.audio.sample_rate if not self.args.encoder_sample_rate else self.args.encoder_sample_rate,
                frame_length=self.config.audio.win_length,
                win_length=self.config.audio.win_length // 2,
                hop_length=self.config.audio.hop_length,
                pad_mode="reflect",
                center=False,
                n_thresholds=100,
                beta_parameters=(2, 18),
                boltzmann_parameter=2,
                resolution=0.1,
                max_transition_rate=35.92,
                switch_prob=0.01,
                no_trough_prob=0.01,
            )
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
            emo_vectors = [vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
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
                center=False,
            )
            batch["energy"] = mel_to_energy(energy_mel_spec)
            # batch["energy"] = self.energy_scaler(batch["energy"])

        if self.args.use_pitch:
            zero_idxs = batch["pitch"] == 0.0
            pitch_norm = self.pitch_scaler(batch["pitch"])
            pitch_norm[zero_idxs] = 0.0
            batch["pitch"] = pitch_norm[:, :, : batch["energy"].shape[-1]]
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
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
            sampler = self.get_sampler(config, dataset, num_gpus, is_eval)

            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
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
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
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

        # if not config.model_args.encoder_sample_rate:
        #     assert (
        #         upsample_rate == config.audio.hop_length
        #     ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        # else:
        #     encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
        #     effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
        #     assert (
        #         upsample_rate == effective_hop_length
        #     ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

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
        speaker_id: str,
        language_id=None,
        d_vector=None,
        ref_waveform=None,
        emotion_vector=None,
        emotion_id=None,
        pitch_transform=None,
        noise_scale=0.66,
        sdp_noise_scale=0.33,
        denoise_strength=0.08,
        split_batch_sentences=True,
    ):
        # TODO: add language_id
        is_cuda = next(self.parameters()).is_cuda

        # Speaker embeddings
        if speaker_id is not None:
            if self.config.use_d_vector_file:
                d_vector = self.speaker_manager.get_mean_embedding(speaker_id, num_samples=None, randomize=False)
                speaker_id = None
            else:
                speaker_id = id_to_torch(speaker_id, cuda=is_cuda)

        if d_vector is not None:
            d_vector = embedding_to_torch(d_vector, cuda=is_cuda)

        # Emotion embeddings
        if emotion_id is not None:
            if self.config.use_d_vector_file:
                emotion_vector = self.emotion_manager.get_mean_embedding(emotion_id, num_samples=None, randomize=False)
            else:
                emotion_id = id_to_torch(emotion_id, cuda=is_cuda)

        if emotion_vector is not None:
            emotion_vector = embedding_to_torch(emotion_vector, cuda=is_cuda)

        # --> TEXT FORMATTING

        # split to sentences
        if split_batch_sentences:
            # TODO: reformat this
            segmenter = pysbd.Segmenter(language="en", clean=True)
            sentences = segmenter.segment(text)
        else:
            sentences = [text]

        input_tensors = []
        input_lengths = []
        for sentence in sentences:

            # --> TEXT NORMALIZATION & PHONEMIZATION

            input_tensor = np.asarray(
                self.tokenizer.text_to_ids(sentence, language=language_id),
                dtype=np.int32,
            )
            input_tensor = numpy_to_torch(input_tensor, torch.long, cuda=is_cuda)
            input_tensors.append(input_tensor)

            input_length = input_tensor.shape[0]
            input_lengths.append(input_length)

        num_sentences = len(sentences)
        input_lengths = torch.tensor(input_lengths)
        max_len = input_lengths.max().item()
        input_tensors = torch.stack(
            [
                F.pad(x, (0, max_len - x.size(-1)), mode="constant", value=0) if max_len - x.size(-1) > 0 else x
                for x in input_tensors
            ],
            dim=0,
        )

        if is_cuda:
            input_tensors = input_tensors.cuda()
            input_lengths = input_lengths.cuda()

        # --> SYNTHESIZE VOICE

        # Set parameters from arguments
        self.inference_noise_scale = noise_scale
        self.inference_noise_scale_dp = sdp_noise_scale

        outputs = self.inference(
            input_tensors,
            x_lengths=input_lengths,
            aux_input={"d_vectors": d_vector.expand(num_sentences, -1), "speaker_ids": speaker_id, "emotion_vectors": emotion_vector.expand(num_sentences, -1), "emotion_ids": emotion_id},
            pitch_transform=pitch_transform,
        )

        # Denoise the output
        if denoise_strength > 0:
            outputs["model_outputs"] = self.denoiser.forward(
                outputs["model_outputs"].squeeze(1), strength=denoise_strength
            )

        # --> POST-PROCESSING

        model_outputs = []
        wav = None
        upsampling_rate = 1

        if self.args.encoder_sample_rate:
            upsampling_rate = self.config.audio.sample_rate // self.args.encoder_sample_rate

        for k, output in enumerate(outputs["model_outputs"]):
            dur = outputs["durations"][k].sum().item() * self.config.audio.hop_length * upsampling_rate
            wav_ = output[:, : int(dur)].data.cpu().numpy()
            if wav is None:
                wav = wav_
            else:
                zero_pad = np.zeros([1, 10000]) * self.length_scale
                wav = np.concatenate([wav, zero_pad, wav_], 1)

            model_outputs.append(outputs)

        return_dict = {
            "wav": wav,
            "input_tensors": input_tensors,
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


if __name__ == "__main__":
    from TTS.tts.configs.vits_config import VitsConfig

    def _create_inputs(config, batch_size=2, device="cuda"):
        input_dummy = torch.randint(0, 24, (batch_size, 10)).long().to(device)
        input_lengths = torch.randint(2, 11, (batch_size,)).long().to(device)
        input_lengths[-1] = 10
        spec = torch.rand(batch_size, config.audio["fft_size"] // 2 + 1, 30).to(device)
        mel = torch.rand(batch_size, config.audio["num_mels"], 30).to(device)
        spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        spec_lengths[-1] = spec.size(2)
        waveform = torch.rand(batch_size, 1, spec.size(2) * config.audio["hop_length"]).to(device)
        energy = torch.rand(batch_size, 1, 30).to(device)
        pitch = torch.rand(batch_size, 1, 30).to(device)
        spk_emb = torch.rand(batch_size, 512).to(device)
        emo_emb = torch.rand(batch_size, 64).to(device)
        return input_dummy, input_lengths, mel, spec, spec_lengths, waveform, energy, pitch, spk_emb, emo_emb

    config = VitsConfig()
    config.model_args.use_pitch = True
    config.model_args.use_energy_predictor = True
    config.model_args.use_context_encoder = True
    config.model_args.use_d_vector_file = True
    config.model_args.d_vector_dim = 512
    config.model_args.use_emotion_vector_file = True
    config.model_args.emotion_vector_dim = 64
    config.model_args.encoder_sample_rate = 22050
    config.model_args.use_sdp = False
    model = Vits.init_from_config(config)
    model.cuda()

    input_dummy, input_lengths, mel, spec, spec_lengths, waveform, energy, pitch, spk_emb, emo_emb = _create_inputs(
        config
    )
    model.forward(
        x=input_dummy,
        x_lengths=input_lengths,
        y=spec,
        y_lengths=spec_lengths,
        mel=mel,
        waveform=waveform,
        energy=energy,
        pitch=pitch,
        attn_priors=None,
        aux_input={"d_vectors": spk_emb, "emotion_vectors": emo_emb},
        binarize_alignment=False,
    )

    model.inference(
        x=input_dummy, x_lengths=input_lengths, aux_input={"d_vectors": spk_emb, "emotion_vectors": emo_emb}
    )
