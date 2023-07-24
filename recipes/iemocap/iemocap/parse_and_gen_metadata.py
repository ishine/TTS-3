import os
import pandas as pd
from tqdm import tqdm
import re

TS_RE = r"[\d\.]+ \- [\d\.]+"
UTT_RE = r"(\w+)"
BET_RE = r"([a-z]{2,4})"
VAD_RE = r"[\d\.]+, [\d\.]+, [\d\.]+"

base_path = 'IEMOCAP_full_release/Session'
dialogs_path = '/sentences/wav'
texts_path = '/dialog/transcriptions'
emotion_path = '/dialog/EmoEvaluation'

count_err = 0
wav_save = []
txt_save = []
spk_save = []
emo_save = []

def read_text(path):
    file = open(path, 'r')
    count = 0
    text_list = []
    while True:
        count += 1
        line = file.readline()
        if not line:
            break
        else:
            text_list.append((line.strip().split(' ')[0], line.strip().split(':')[-1][1:]))
    file.close()
    return dict(text_list)

for session in range(1,6):
    dialogs = os.listdir(base_path + str(session)+ dialogs_path)
    for dialog in dialogs:
        wavs = os.listdir(base_path + str(session)+ dialogs_path + '/' + dialog)

        texts = base_path + str(session)+ texts_path + '/' + dialog + '.txt'
        utt2txt = read_text(texts)

        emos = base_path + str(session)+ emotion_path + '/' + dialog + '.txt'
        with open(emos.strip()) as f:
            txt = f.read()
        utt2emo = dict(re.findall(rf"\[{TS_RE}\]\s{UTT_RE}\s{BET_RE}\s\[{VAD_RE}\]", txt))

        for wav in wavs:
            if '.wav' in wav:
                wav_save.append(base_path + str(session)+ dialogs_path + '/' + dialog + '/' + wav)
                txt_save.append(utt2txt[wav[:-4]])
                if 'F' in wav.split('_')[-1]:
                    spk_save.append('F'+str(session))
                elif 'M' in wav.split('_')[-1]:
                    spk_save.append('M'+str(session))
                emo_save.append(utt2emo[wav[:-4]])
            else:
                count_err += 1
                continue


df = pd.DataFrame({'Wavs':wav_save, 'Text':txt_save, 'Speaker':spk_save, 'Emotion': emo_save})

# Filter Undecisive Emotion
df = df[(df.Emotion != 'xxx')]
df = df[(df.Emotion != 'oth')]
df.reset_index(inplace=True,drop=True)

if not os.path.exists('files'):
    os.mkdir('files')

for i in tqdm(range(len(df))):
    spk = df.loc[i, "Speaker"]
    emo = df.loc[i, "Emotion"]
    wav = df.loc[i, "Wavs"]

    new_dir = spk + '_' + emo + '/'
    if not os.path.exists('files/'+new_dir):
        os.makedirs('files/'+new_dir)
    os.rename(wav, 'files/'+new_dir+wav.split('/')[-1])

    df.loc[i, "Wavs"] = new_dir+wav.split('/')[-1]


df.to_csv(path_or_buf = 'metadata_all.csv', sep='|', index=False, header=False)