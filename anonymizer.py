from typing import Union

import os

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from encodec.utils import convert_audio

from TTS.tts.layers.bark.inference_funcs import semantic_tokens_from_audio, load_voice
import torch
import torchaudio
from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer

from vocos import Vocos
from char_vocos.pretrained import CharVocos
from utils import get_trained_vocos

from speechbrain.inference.ASR import EncoderASR

class Anonymizer(torch.nn.Module):
    def __init__(self, checkpoint_dir: str, voice_dirs: Union[list[str], None] = None, use_vocos=True,
                 vocos_checkpoint=None, device='cpu'):
        super().__init__()
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} not found, creating it")
            os.makedirs(checkpoint_dir)

        # 0. get speechbrain asr
        self.asr_model = EncoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-librispeech",
            savedir="pretrained_models_sb/asr-wav2vec2-librispeech",
            run_opts={"device": device, "freeze": True}
        )

        # 1. initialize Bark
        config = BarkConfig()  # don't change the custom config for the love of god
        self.model = Bark.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)
        # self.model.to('cuda')

        # 2. initialize the awesome, bark-distilled, unlikely-yet-functioning audio tokenizer
        hubert_manager = HubertManager()
        hubert_manager.make_sure_tokenizer_installed(model_path=self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"])
        self.hubert_model = CustomHubert(
            checkpoint_path=self.model.config.LOCAL_MODEL_PATHS["hubert"])  # .to(self.model.device)
        self.tokenizer = HubertTokenizer.load_from_checkpoint(
            self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"], map_location=self.model.device
        )

        self.voice_dirs = voice_dirs
        self.sample_rate = self.model.config.sample_rate

        self.use_vocos = use_vocos
        if self.use_vocos:
            # 3. if setting is given, initialize culos, i mean vocos
            if vocos_checkpoint is None:
                print("Using normal pretrained vocos")
                self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
            elif type(vocos_checkpoint) is str:
                print(f"Using vocos checkpoint {vocos_checkpoint}")
                self.vocos = get_trained_vocos(vocos_checkpoint)
            elif type(vocos_checkpoint) is tuple:
                config_path, model_path = vocos_checkpoint
                print(f"Using char vocos\n\tconfig_path: {config_path}\n\tmodel_path: {model_path}")
                self.vocos = CharVocos.from_local_pretrained(config_path, model_path)
            else:
                raise ValueError("Argument 'vocos_checkpoint' can only be str (path to a trained checkpoint) or None.")

    def forward(
            self,
            audio: Union[torch.Tensor, str],
            target_voice_id: str = 'random',
            coarse_temperature: float = 0.7
    ):
        # You can give the audio as path to a wav. In this case, resampling and reshaping is done
        # If you directly give a tensor: must be 1 channel, 24k sr, and shape (1, L)
        # batched inference is currently not supported, sorry
        if isinstance(audio, str):
            audio_path = audio
            audio, sr = torchaudio.load(audio_path)
            audio = convert_audio(audio, sr, self.model.config.sample_rate, self.model.encodec.channels)
            audio = audio.to(
                self.model.device)  # there used to be an unsqueeze here but then they squeeze it back so it's useless

            audio_asr_model = self.asr_model.load_audio(audio_path, savedir='./useless_links').unsqueeze(
                0)  # will automatically resample
            audio_asr_model = audio_asr_model.to(self.model.device)
        else:
            raise ValueError('In this implementation, audio must only be provided in the form of a string.')

        fake_lens = torch.tensor([1.0], device=self.model.device)
        log_softmax_scores = self.asr_model.encode_batch(audio_asr_model, fake_lens)  # recall, this is log softmax (1, T, 31)
        # log_softmax_scores = log_softmax_scores.squeeze()

        # 1. Extraction of semantic tokens
        semantic_vectors = self.hubert_model.forward(audio, input_sample_hz=self.model.config.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors)
        semantic_tokens = semantic_tokens.cpu().numpy()  # they must be shifted to cpu
        # this probably slows things down, but the following api function from bark specifically requires numpy
        # but i mean, what the fuck do i know

        # 2. Load voice as a history prompt as a tuple (semantic_prompt, coarse_prompt, fine_prompt)
        if not self.voice_dirs:
            assert target_voice_id == 'random', """If no voice dirs are given, the target voice must be 'random'.
            Note that, regardless of this, 'random' always means 'use an empty semantic and coarse prompts'.
            So even if target_voice_id == 'random', the voice_dirs will be ignored (it does NOT mean it will pick a
            random voice from there).
            ...this should probably go into some documentation. Why am I writing it here?"""
        history_prompt = load_voice(self.model, target_voice_id, self.voice_dirs)

        # 3. Regression of acoustic tokens with bark api
        # 'temp' here is only the coarse temperature. The fine temperature is internally fixed to 0.5
        # (i fiddled with it a bit and it does seem a bit of a sweet spot, any higher and the audio gets a bit dirty)
        # the other two returned values are coarse and fine tokens, we don't need them for now

        # audio_arr_encodec, _, _ = self.model.semantic_to_waveform(
        #     semantic_tokens, history_prompt=history_prompt, temp=coarse_temperature
        # )

        x_coarse_gen, x_fine_gen = self.model.semantic_to_fine(
            semantic_tokens, history_prompt=history_prompt, coarse_temp=coarse_temperature
        )
        x_fine_gen = torch.tensor(x_fine_gen, device=self.model.device)
        features = self.vocos.codes_to_features(x_fine_gen)
        # i'm keeping the bandwidth the same as encodec (6 kbps)
        # i genuinely have no idea what difference it makes inside vocos
        # it seems to only internally affect layer norms, where they train the scale/bias according to the bitrate
        # do they know that torch's layer norm already scales? I guess they do.
        # yeah, the index '2' is 6 kbps according to their git
        bandwidth_id = torch.tensor([2], device=self.model.device)
        # audio_arr_vocos = self.vocos.decode(features, bandwidth_id=bandwidth_id)
        audio_arr_vocos = self.vocos.decode(features, bandwidth_id=bandwidth_id, log_probs_transcript=log_softmax_scores)
        # return audio_arr_encodec, audio_arr_vocos
        return audio_arr_vocos


checkpoint_dir = '/homes/panariel/.local/share/tts/tts_models--multilingual--multi-dataset--bark'
# checkpoint_dir = 'pretrained_models_dumpster/tts_models--multilingual--multi-dataset--bark'  # this also works

# anonymizer = Anonymizer(checkpoint_dir)
# anonymizer.to('cuda')
#
# print('Done initializing')
# print(f'\tBark is on {anonymizer.model.device}')
# print(f'\tCustomHubert is on {anonymizer.hubert_model.model.feature_projection.projection.weight.device}')
# print(
#     f'\tHubertTokenizer is on {anonymizer.tokenizer.fc.weight.device} (and btw it\'s version {anonymizer.tokenizer.version})')
