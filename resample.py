import soundfile as sf
import glob
import os
import librosa
from argparse import ArgumentParser
import tqdm

parser = ArgumentParser(description="Resample a wav folder to 16Khz.")
parser.add_argument('in_fold', help='Which is the input folder, will take all wavs from it.')
parser.add_argument('out_fold', help='Where to place the new wavs. Will create it if doesn\'t exist.')
parser.add_argument('--extension', default='wav', help='Which kind of files do you want? wav? flac? mp3? your mom?')
parser.add_argument('--ds_kind', help='Which dataset kind is this?', default='libri_dev')
args = parser.parse_args()

accepted_modes = ['libri360', 'libri_dev', 'libri_test', 'vctk']
assert args.ds_kind in accepted_modes, f"ds_king must be in {accepted_modes} (got {args.ds_kind})"

TARGET_SR = 16000

# create output folder if not exists
out_fold = args.out_fold
if not os.path.isdir(out_fold):
    print(f"{out_fold} does not exist, creating it.")
    os.makedirs(out_fold)

# list all files of the desired extension
pattern = os.path.join(args.in_fold, f'*.{args.extension}')
in_paths = glob.glob(pattern)

out_paths = [os.path.join(out_fold, os.path.basename(in_path)) for in_path in in_paths]

assert len(in_paths) == len(out_paths)  # better safe than sorry
num_files = len(in_paths)

for i, in_single_path in enumerate(in_paths, 1):
    # here the madness begins
    if args.ds_kind == 'libri360' or args.ds_kind == 'libri_test':
        # just put the files in the output folder
        out_single_path = os.path.join(out_fold, os.path.basename(in_single_path))
    elif args.ds_kind == 'libri_dev':
        # each utterance has its own subfolder
        filename = os.path.basename(in_single_path)
        utt_id, _ = os.path.splitext(filename)
        individual_fold_path = os.path.join(out_fold, utt_id)
        os.mkdir(individual_fold_path)
        out_single_path = os.path.join(individual_fold_path, filename)
    elif args.ds_kind == 'vctk':
        # each speaker has its own subfolder
        speaker_id = os.path.basename(in_single_path).split('_')[0]
        individual_fold_path = os.path.join(out_fold, speaker_id)
        if not os.path.isdir(individual_fold_path):
            os.mkdir(individual_fold_path)
        out_single_path = os.path.join(individual_fold_path, os.path.basename(in_single_path))
    else:
        raise ValueError("What? A wrong ds_type parameter reached here.")

    end = '\n' if i == num_files else '\r'
    print(f"[{i}/{num_files}] Reading {in_single_path}, reampling to {TARGET_SR} Hz, saving to {out_single_path}")
    wav, sr = librosa.load(in_single_path, sr=TARGET_SR)
    sf.write(out_single_path, wav, samplerate=sr)