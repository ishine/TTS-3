# Preserving spoken content in voice anonymisation with character-level vocoder conditioning
Author's implementation of [Preserving spoken content in voice anonymisation with character-level vocoder conditioning](https://arxiv.org/abs/link_not_available_yet_google_the_paper_name_in_the_meantime), published at SPSC Symposium 2024. The repository provides model weights and instructions on how to perform inference on the [VoicePrivacy Challenge 2024](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024) datasets.

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
For this version of the framework, you also need [Vocos](https://github.com/gemelo-ai/vocos).
```
pip install vocos
```

## Files and folders
- the model is class is in `anonymizer.py`
- `data.py` has a dataset class that also handles the speaker-lever/utterance-level mapping
- `inference.py` performs anonymization of an entire data partition with multi-GPU processing (uses huggingface accelerate)
- `simple_forward.py` shows a no-frills forward call to the model, anonymizing one utterance with one of the available target voices
- `run_inference_eval_2024.sh` and `run_inference_train360_2024.sh` are scripts with some pre-defined parameters to perform anonymization specifically of the VoicePrivacy 2024 datasets
- `resample.py` is a utility to resample a wav folder from 24 kHz (model output) to 16 kHz (required by VoicePrivacy Challenge)
- `bark_patch.patch` is a git patch that contains all the modifications that were made to the 🐸TTS Bark modules to adapt them for anonymization. It technically allows to recreate the model from a fresh 🐸TTS fork - though it's not really needed for anything here
- `speaker_mappings` contains the generated randomly speaker-to-pseudospeaker mappings used to produce the results reported in the paper
- `suno_voices` contains the voice samples made available by Suno AI, developers of Bark. They are taken from [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)
- the `char_vocos` folder contains the implementation of the modified version of Vocos with the character-level conditioning (an actual contribution in model code form? Really?! woooow). Keep in mind that to instantiate it you need the `char_vocos_config.yaml` file.

## Usage
### Using the model off-the-shelf
EnCodec natively operates at 24 kHz; therefore, so does Bark; therefore, so does our model. If you forward an audio array, make sure it is 24k and of shape (1,L). You can also just provide a file path, in which case the model will internally handle any needed resampling.  
Batched inference is not supported. I know, that sucks. I didn't dare tinkering with Bark's internal modules to make them support batched input - not yet, at least.

Note that this version uses a re-trained version of Vocos that uses character-level conditioning. [➡️ HERE IS THE LINK TO DOWNLOAD THE CHECKPOINT ⬅️.](https://nextcloud.eurecom.fr/s/xoHw6fjgJBJtWna).
We do not provide the training code because, trust me, you don't wanna see that abomination. But if you are really itching to try and train the character-conditioned Vocos by yourself, send an email to the first author.

Once you have the file `char_vocos.ckpt`, you supply it to the class constructor along with the configuration file `char_vocos_config.yaml`. They need to be joined in a tuple in which the config file goes first. Not a list, not a set, a tuple. Not doing so will result in the model using the normal EnCodec decoder, collapsing back to the anon system in the other branch.
You can provide pseudo-speaker IDs to the forward call. To do so, when instantiating the model, pass the position of the voice prompts folder (in this repo, it is `suno_voices/v2`). To select a pseudo-speaker, pass a target voice id to the forward method. The IDs of the provided voice prompts have the format `<country code>_speaker_<number>` (note that not passing a voice dir will result in the model sampling the acoustic prompt from nothing - see the other branch for more info).
```python
import os
from anonymizer import Anonymizer

chekpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')
voices = 'suno_voices/v2'

anonymizer = Anonymizer(args.checkpoint_dir, voice_dirs=[voices], vocos_checkpoint=('char_vocos_config.yaml', 'char_vocos.ckpt'))
anon_wav = anonymizer('path_to_audio.wav', target_voice_id='it_speaker_0')
```

### Anonymizing the VoicePrivacy 2024 datasets
Scripts `run_inference_eval_2024.sh` and `run_inference_train360_2024.sh` are available to perform the anonymization of the VoicePrivacy Challenge 2024 datasets.

For example, to anonymize libri dev enrolls, run
```bash
bash run_inference_eval_2024.sh libri_dev enrolls <root of the data that will be contatenated to the paths in the SCP file>
```
For libri 360, you need to modify the root internally, because basically I didn't have the time to make it as modular as the version in the old branch. Mea culpa, life is hard, academia is harder.

### Final remark
Yes, this README kinda sucks compared to the one in the aforementioned "other branch". Hopefully I'll be able to refine it a little bit afterwards. Bye.
