# encoding=utf-8

import pickle

seq_set = set()
with open('../data/NA12878.pcr_MSssI.r9.timp.081016.pass.nanopolish.cpgfeatures.tsv', 'r') as file:
    for line in file.readlines()[1:]:
        columns = line.split()
        for i in range(6):
            seq = columns[4 + i * 3]
            seq_set.add(seq)

print(len(seq_set))

with open('../data/NA12878.pcr.r9.timp.081016.pass.nanopolish.cpgfeatures.tsv', 'r') as file:
    for line in file.readlines()[1:]:
        columns = line.split()
        for i in range(6):
            seq = columns[4 + i * 3]
            seq_set.add(seq)

print(len(seq_set))

seq_dict = dict()
for i, seq in enumerate(sorted(seq_set)):
    seq_dict[seq] = i

print(seq_dict)
print(len(seq_dict))

with open('../cache/seq_dict.pkl', 'wb') as file:
    pickle.dump(seq_dict, file)
