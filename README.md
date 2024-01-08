# Speaker anonymization language model etc

## Installation
Clone this repo (duh), create a new conda env, then pip install locally the library

```
git clone https://github.com/m-pana/spk_anon_nac_lm.git
cd spk_anon_nac_lm
conda create -n spk_anon python=3.11
conda activate spk_anon
pip install -e coqui_tts
```
done.

You might run into some problems while doing this. Coqui TTS has [this](https://github.com/pypa/pip/issues/12305) problem of recursing into pip.  
For me, this only worked by installing with python 3.11 (3.8, 3.9 and 3.10 will all fail).
Another issue that was specific to my setup is that pip automatically collected torch 2.1.2, which was too new for my cuda version (11.2). Thus, I had to downgrade to 2.0.2 (and downgrade torchaudio and torchvision accordingly):
```
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

## Usage
### Just a forward
To anonymize a single 24k utterance with a totally random seed. Instantiate model with pretrained checkpoint directory, which will normally be `~/.local/share/tts/tts_models--multilingual--multi-dataset--bark`
```
import os  # literally just to expand ~ to $HOME
from anonymizer import Anonymizer

chekpoint_dir = os.path.expanduser('~/.local/share/tts/tts_models--multilingual--multi-dataset--bark')

anonymizer = Anonymizer(args.checkpoint_dir)  # you can call .to(device) if you want
anon_wav = anonymizer('path_to_audio.wav')  #output has shape (L,)
```
### Using custom voices
Instantiate the model with a folder of voices. (provided in the repo)
```
# same as before, then

voices = 'suno_voices/v2'
anonymizer = Anonymizer(args.checkpoint_dir, voice_dirs=[voices])
anon_wav = anonymizer('path_to_audio.wav', target_voice_id='it_speaker_0')
```
# other stuff
- VPC2022 repo must be cloned within the folder, else you can modify the `data_root` path in the bash scripts
- must change the port in accelerate config every time (or set it to 0, but didn't work for me)