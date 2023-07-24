import pandas as pd


df = pd.read_csv('metadata_all.csv', sep='|', names=["paths", "text", "speaker", "style"])

from sklearn.utils import shuffle
df_shuffled = shuffle(df)
a = round(len(df_shuffled)*0.98)
df_train = df_shuffled.iloc[0:a]

df_valtest = df_shuffled.iloc[a:]

print("Valid:")
print(df_valtest)

df_train.to_csv('./metadata_train_' + 'all' + '.csv', index=False, header=False, sep='|')
df_valtest.to_csv('./metadata_val_' + 'all' + '.csv', index=False, header=False, sep='|')