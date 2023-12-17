import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Metadata_type")
parser.add_argument('-type', action='store', dest='type', type=str, default="Neutral", required=True, help="Partition of the dataset to consider, can be a specific style or speaker")


args = parser.parse_args()

names = []
sep = "|"
files_folder = './files'

file = open('transcriptions.txt')
content = file.readlines()

for line in content:
    info = line.split('\t')
    info[-1] = info[-1].split('\n')[0]
    spk = info[0].split('_')[0]
    wav = info[0].split('_')[1]
    text = info[1]
    style = info[2]
    path = spk + '_' + style + '/' + spk + '_' + wav + '.wav'
    write = path + sep + text + sep + spk + sep + style

    if args.type in ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprise']:
        if style == args.type: 
           names.append(write)
        else:
           pass
    elif args.type in ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']:
        if spk == args.type: 
           names.append(write)
        else:
           pass
    elif args.type == 'all':
        names.append(write)
    else:
        raise NotImplementedError

with open('metadata_' + args.type + '.csv', 'w') as f:
    for line in names:
        f.write(f"{line}\n")
