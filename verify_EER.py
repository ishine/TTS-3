"""
Compute EER of original data, encodec data, vocos data in one fell swoop.
Also, ponder about the misery of human condition.
"""

import argparse
import os

from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.utils.edit_distance import wer_details_for_batch, wer_summary
from speechbrain.dataio.wer import print_wer_summary
import torchaudio

def main():
    parser = argparse.ArgumentParser(description="A cute script to handle command-line arguments!")
    parser.add_argument('anon_data', metavar='anon_data_folder', type=str, help='Folder containing anonymous data')
    parser.add_argument('origin_data', metavar='origin_data_folder', type=str, help='Folder containing original data')
    parser.add_argument('text_file', type=str, help='File with the transcription kaldi style fuck')
    parser.add_argument('--device', type=str, default='cuda', help='File with the transcription kaldi style fuck')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Do something adorable with the arguments here!
    print('A SpicBrein is born?')
    # asr_model = EncoderASR.from_hparams(
    #     source="speechbrain/asr-wav2vec2-librispeech",
    #     savedir="pretrained_models_sb/asr-wav2vec2-librispeech",
    #     run_opts={"device": args.device, "freeze": True}
    # )

    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
        run_opts={"device": args.device, "freeze": True}
    )

    data = {}
    wer_details = [[], [], []]  # original, encodec, vocos
    with open(args.text_file) as file:
        i = 1
        for line in file:
            print('Next file.')
            # Split the line into utt_id and ground_truth_text
            utt_id, ground_truth_text = line.strip().split(' ', 1)
            data[utt_id] = {'ground_truth': ground_truth_text}

            # use the utt_id to grab: original wav, encodec wav, vocos wav, cazzoinculo wav
            origin_path = os.path.join(args.origin_data, utt_id, f'{utt_id}.wav')
            origin_wav, sr_orig = torchaudio.load(origin_path)

            encodec_path = os.path.join(args.anon_data, f'{utt_id}_encodec.wav')
            vocos_path = os.path.join(args.anon_data, f'{utt_id}_vocos.wav')

            try:
                encodec_wav, sr_enco = torchaudio.load(encodec_path)
                vocos_wav, sr_voco = torchaudio.load(vocos_path)
            except RuntimeError as re:
                if 'Error loading audio file' in str(re):
                    print(f'Utterance {utt_id} not found in text list, ignoring.')
                    continue
                else:
                    print(f'WTF just happened? I got:\n{re}\nCrashing everything.')
                    quit()

            # here we just assume we have both encodec and vocos wav
            origin_wav = asr_model.audio_normalizer(origin_wav.squeeze(), sample_rate=sr_orig)
            encodec_wav = asr_model.audio_normalizer(encodec_wav.squeeze(), sample_rate=sr_enco)
            vocos_wav = asr_model.audio_normalizer(vocos_wav.squeeze(), sample_rate=sr_voco)

            wavs, lengths = batch_pad_right([origin_wav, encodec_wav, vocos_wav])
            transcripts, _ = asr_model.transcribe_batch(wavs, wav_lens=lengths)
            data[utt_id].update({
                'transcript_original': transcripts[0],
                'transcript_encodec': transcripts[1],
                'transcript_vocos': transcripts[2]})
            robe = '\n'.join([f'\t{k}: {v}' for k, v in data[utt_id].items()])

            print(f'[{i}/boh] {utt_id}\n', robe)
            i += 1

            for tr, detail_list in zip(transcripts, wer_details):
                utt_details = wer_details_for_batch(ids=[utt_id], refs=[tr.split()], hyps=[ground_truth_text.split()])
                detail_list.extend(utt_details)

            # if i == 10:
            #     print('Breaking at 10 iterations')
            #     break

    # loop is over, fuck my life
    wer_summaries = [wer_summary(dt) for dt in wer_details]

    for summary, name in zip(wer_summaries, ['original', 'encodec', 'vocos']):
        print(f'WER of data {name}')
        print_wer_summary(summary)
        print('------------------')



if __name__ == "__main__":
    main()
