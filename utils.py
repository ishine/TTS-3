import torch
import torchaudio
from vocos import Vocos
from huggingface_hub import hf_hub_download
import soundfile as sf

# repo_id = "charactr/vocos-encodec-24khz"
# hparams_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
# culos = Vocos.from_hparams(hparams_path)
#
# print(culos)
# print('Ciao')


# checkpoint_train_path = "/medias/speech/projects/panariel/train_vocos/vocos/logs/discriminator_v_3/checkpoints/last.ckpt"
# d = torch.load(checkpoint_train_path)
# state_dict = d['state_dict']
# # abominio
# filtered_state_dict = {k: state_dict[k] for k in state_dict.keys() if not ('multiresddisc' in k or 'multiperioddisc' in k or 'melspec_loss' in k)}
# missing, unexpected = culos.load_state_dict(filtered_state_dict)
# print(f'Missing {missing}')
# print(f'r/Unexpected {unexpected}')
# print('Copy-synthesizing your mom')
# wav, sr = torchaudio.load('/medias/speech/projects/panariel/train_vocos/98-121658-0017.flac')
# print(wav.shape)
# wav_res = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)
# print(wav_res.shape)
# cs = culos(wav_res, bandwidth_id=torch.tensor([2]))
# print(cs.shape)
# sf.write('cp.wav', cs.squeeze(), samplerate=24000)
# print('Fatto')


def get_trained_vocos(checkpoint_train_path) -> Vocos:
    # instantiate new copy of network
    repo_id = "charactr/vocos-encodec-24khz"
    hparams_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    vaffanculos = Vocos.from_hparams(hparams_path)

    # load the saved state dict and only keep those related to the generator (discrim is useless)
    d = torch.load(checkpoint_train_path)
    state_dict = d['state_dict']
    # dict comprehension abomination
    filtered_state_dict = {k: state_dict[k] for k in state_dict.keys() if
                           not ('multiresddisc' in k or 'multiperioddisc' in k or 'melspec_loss' in k)}
    missing_keys, unexpected_keys = vaffanculos.load_state_dict(filtered_state_dict)
    print(f'Loaded Vocos (encodec) from checkpoint in path {checkpoint_train_path}.\n\tMissing keys: {missing_keys}\n'
          f'\tUnexpected keys: {unexpected_keys}')
    vaffanculos.eval()
    return vaffanculos
