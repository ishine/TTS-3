import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser(description="Metadata_type")
parser.add_argument('-type', action='store', dest='type', type=str, default="Neutral", required=True, help="Partition of the dataset to consider, can be a specific style or speaker")

args = parser.parse_args()

df_train = pd.read_csv('metadata_train_all.csv', sep='|', names=["paths", "text", "speaker", "style"])
df_val = pd.read_csv('metadata_val_all.csv', sep='|', names=["paths", "text", "speaker", "style"])

# Filter Train
if args.type in df_train['style'].unique():
    df_train = df_train[df_train['style']==args.type]
elif args.type in df_train['speaker'].unique():
    df_train = df_train[df_train['speaker']==args.type]
else:
    raise NotImplementedError

# Filter Val
if args.type in df_val['style'].unique():
    df_val = df_val[df_val['style']==args.type]
elif args.type in df_val['speaker'].unique():
    df_val = df_val[df_val['speaker']==args.type]
else:
    raise NotImplementedError

print(df_train)
print(df_val)

df_train.to_csv('./metadata_train_' + args.type + '.csv', index=False, header=False, sep='|')
df_val.to_csv('./metadata_val_' + args.type + '.csv', index=False, header=False, sep='|')