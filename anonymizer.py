from typing import Union

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

class Anonymizer(torch.nn.Module):
    def __init__(self, checkpoint_dir: str, voice_dirs: Union[list[str], None] = None, use_vocos=True):
        super().__init__()
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} not found, creating it")
            os.makedirs(checkpoint_dir)

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
            self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

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
            audio, sr = torchaudio.load(audio)
            audio = convert_audio(audio, sr, self.model.config.sample_rate, self.model.encodec.channels)
            audio = audio.to(
                self.model.device)  # there used to be an unsqueeze here but then they squeeze it back so it's useless

        # 1. Extraction of semantic tokens
        semantic_vectors = self.hubert_model.forward(audio, input_sample_hz=self.model.config.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors)
        semantic_tokens = semantic_tokens.cpu().numpy() # they must be shifted to cpu
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
        if not self.use_vocos:
            audio_arr, x_coarse_gen, x_fine_gen = self.model.semantic_to_waveform(
                semantic_tokens, history_prompt=history_prompt, temp=coarse_temperature
            )
        else:
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
            audio_arr = self.vocos.decode(features, bandwidth_id=bandwidth_id)
        return audio_arr, x_coarse_gen, x_fine_gen


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
