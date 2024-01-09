# Speaker anonymization using neural audio codec language models
Author's implementation of [Speaker anonymization using neural audio codec language models](https://arxiv.org/abs/2309.14129), published at ICASSP 2024. The repository provides model weights and instructions on how to perform inference on the [VoicePrivacy Challenge 2022](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) dataset.

## Installation
The system is basically a modified version of the Bark model as implemented in the 🐸TTS library ([see here](https://docs.coqui.ai/en/dev/models/bark.html)). I had to tweak some source files of the library, so the main dependecy is a forked version of 🐸TTS that you have to install directly from this repository. To do so:
1. Clone this repo (duh!)
2. Create a new conda environment (or another kind of virtual env of your liking) with Python 3.11
3. pip-install the modified libray from the files contained in `coqui_tts`

```bash
git clone https://github.com/m-pana/spk_anon_nac_lm.git
cd spk_anon_nac_lm
conda create -n spk_anon python=3.11
conda activate spk_anon
pip install -e coqui_tts
```
And you should be done.

### Troubleshooting
Below are a few problems I ran into when installing, I'm reporting them in case they are useful to others.

1. Pip gave me a `ResolutionTooDeep: 200000` error when re-installing my forked version of 🐸TTS the first few times. The same issue [has been reported by others](https://github.com/pypa/pip/issues/12305) in the pip repo for 🐸TTS.  
For me, what made it work was making sure to install with Python 3.11. Versions 3.8, 3.9 and 3.10 all failed in similar ways.

2. I believe 🐸TTS always tries to install the most recent version of PyTorch. At the time of writing, it will attempt to install version 2.1.2. I am running CUDA 11.2, which is apparently too obsolete for it. If you are in the same situation, you can downgrade to PyTorch 2.0.1 (in fact, this is how I ran my experiments). I recommend adjusting TorchAudio and TorchVision accordingly:
```bash
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```
(I don't think downgrading torchvision is actually necessary, but the error message is scary so I did it anyway)

## Files and folders
- the model is class is in `anonymizer.py`
- `data.py` has a dataset class that also handles the speaker-lever/utterance-level mapping
- `inference.py` performs anonymization of an entire data partition with multi-GPU processing (uses huggingface accelerate)
- `simple_forward.py` shows a no-frills forward call to the model, anonymizing one utterance with one of the available target voices
- `run_inference_eval.sh` and `run_inference_train360.sh` are scripts with some pre-defined parameters to perform anonymization specifically of the VoicePrivacy datasets
- `bark_patch.patch` is a git patch that contains all the modifications that were made to the 🐸TTS Bark modules to adapt them for anonymization. It technically allows to recreate the model from a fresh 🐸TTS fork - though it's not really needed for anything here
- `speaker_mappings` contains the generated randomly speaker-to-pseudospeaker mappings used to produce the results reported in the paper
- `suno_voices` contains the voice samples made available by Suno AI, developers of Bark. They are taken from [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)

## Usage
### Just a forward
To anonymize a single 24k utterance with a totally random seed. Instantiate model with pretrained checkpoint directory, which will normally be `~/.local/share/tts/tts_models--multilingual--multi-dataset--bark`
```python
import os  # just needed to expand '~' to $HOME
from anonymizer import Anonymizer

chekpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')

anonymizer = Anonymizer(args.checkpoint_dir)  # you can call .to(device) if you want
anon_wav = anonymizer('path_to_audio.wav')  #output has shape (L,)
```
### Using custom voices
Instantiate the model with a folder of voices. (provided in the repo)
```python
# same as before, then

voices = 'suno_voices/v2'
anonymizer = Anonymizer(args.checkpoint_dir, voice_dirs=[voices])
anon_wav = anonymizer('path_to_audio.wav', target_voice_id='it_speaker_0')
```
# other stuff
- VPC2022 repo must be cloned within the folder, else you can modify the `data_root` path in the bash scripts
- must change the port in accelerate config every time (or set it to 0, but didn't work for me)