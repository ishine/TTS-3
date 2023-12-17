import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

parser = ArgumentParser(description="Metadata_type")
parser.add_argument('-type', action='store', dest='type', type=str, default="Neutral", required=True, help="Partition of the dataset to consider, can be a specific style or speaker")

args = parser.parse_args()

df = pd.read_csv('metadata_' + args.type + '.csv', sep='|', names=["paths", "text", "speaker", "style"])
paths = df['paths']

with open('./perceptual_references.txt', 'r') as f:
    refs = f.readlines()
refs = [item[:-1] for item in refs]
train_idxs = []
val_idxs = []
for idx in range(len(paths)):
    path = paths[idx]
    path = path.split('/')[1].replace('_','/')[:-4]
    if path in refs:
         val_idxs.append(idx)
    else:
         train_idxs.append(idx)

df_val = df.loc[val_idxs]
df_train = df.loc[train_idxs]

df_train.to_csv('./metadata_train_' + args.type + '.csv', index=False, header=False, sep='|')
df_val.to_csv('./metadata_val_' + args.type + '.csv', index=False, header=False, sep='|')

