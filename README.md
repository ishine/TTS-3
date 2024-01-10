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
- `resample.py` is a utility to resample a wav folder from 24 kHz (model output) to 16 kHz (required by VoicePrivacy Challenge)
- `bark_patch.patch` is a git patch that contains all the modifications that were made to the 🐸TTS Bark modules to adapt them for anonymization. It technically allows to recreate the model from a fresh 🐸TTS fork - though it's not really needed for anything here
- `speaker_mappings` contains the generated randomly speaker-to-pseudospeaker mappings used to produce the results reported in the paper
- `suno_voices` contains the voice samples made available by Suno AI, developers of Bark. They are taken from [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)


## Usage
### Using the model off-the-shelf
EnCodec natively operates at 24 kHz; therefore, so does Bark; therefore, so does our model. If you forward an audio array, make sure it is 24k and of shape (1,L). You can also just provide a file path, in which case the model will internally handle any needed resampling.  
Batched inference is not supported. I know, that sucks. I didn't dare tinkering with Bark's internal modules to make them support batched input - not yet, at least.

Anyhow, below is the code to anonymize a single wav file with a totally random pseudo-voice. When instantiating the model, you need to pass the directory where the pretrained weights are stored; normally, it will be `~/.local/share/tts/tts_models--multilingual--multi-dataset--bark`.
```python
import os  # just needed to expand '~' to $HOME
from anonymizer import Anonymizer

chekpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')

anonymizer = Anonymizer(args.checkpoint_dir)  # you can call .to(device) if you want
anon_wav = anonymizer('path_to_audio.wav')  # output has shape (L,)
```
To perform speaker-level anonymization, you need to provide a pseudo-speaker prompt for each speaker. To do so, when instantiating the model, pass the position of the voice prompts folder (in this repo, it is `suno_voices/v2`). To select a pseudo-speaker, pass a target voice id to the forward method. The IDs of the provided voice prompts have the format `<country code>_speaker_<number>`.
```python
import os
from anonymizer import Anonymizer

chekpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')
voices = 'suno_voices/v2'

anonymizer = Anonymizer(args.checkpoint_dir, voice_dirs=[voices])
anon_wav = anonymizer('path_to_audio.wav', target_voice_id='it_speaker_0')
```

### Anonymizing the VoicePrivacy 2022 datasets
Scripts `run_inference_eval.sh` and `run_inference_train360.sh` are available to perform the anonymization of the VoicePrivacy Challenge 2022 datasets (respecting the spk-level/utt-level rules). To do that, you will of course need to download the datasets first. The scripts assume that the top-level folder of the [challenge repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) has been placed in the root of this repo. If not, you can change the parameter `data_root` within the scripts.

For example, to anonymize libri dev enrolls, run
```bash
bash run_inference_eval.sh libri_dev enrolls
```
Similarly, for libri360:
```bash
bash run_inference_train360.sh 1
```
The first and only positional parameter can be `{1,2,3,4}`. I split the file list in 4 parts as a sloppy way to make up for the lack of batched inference: at least you can run more multiple processes in parallel on different parts of libri360 to speed things up. It's unbelievably stupid, but I wanna see you come up with something better when the conference deadline is in a week.  
If you decide to comply with my hacky design and launch multiple scripts in parallel, you should use a different process port for Accelerate in every run - otherwise anything but the first script you launch will crash. To avoid such a tragic outcome, modify the `main_process_port` value in `accelerate_config.yaml` before running a new process (I think if you leave the value `null` it will just select the next available port, but I haven't tested that).  
The `accelerate_config.yaml` file in this repository is set to run the inference on 4 GPUs in parallel.

## Other comments
- There is no training code because there was no training. This model is basically the result of bypassing the semantic regressor of Bark and using ground-truth semantic tokens instead of estimating them from text input. Thus, all modules are taken from either Bark or EnCodec.
- I do not have the code of the alternative Speechbrain-based evaluation pipeline - if you are interested in that, please contact the second author of the paper.