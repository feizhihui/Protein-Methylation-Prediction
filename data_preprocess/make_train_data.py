# encoding=utf-8

import pickle

with open('../cache/seq_dict.pkl', 'rb') as file:
    seq_dict = pickle.load(file)


def load_data(file_path):
    train_seq_data = []
    train_prop1_data = []
    train_prop2_data = []
    with open(file_path, 'r') as file:
        for line in file.readlines()[1:]:
            columns = line.split()
            seqs = []
            for i in range(6):
                seq = columns[5 + i * 4]
                seqs.append(seq_dict[seq])
            train_seq_data.append(seqs)
            props = []
            for i in range(6):
                prop = columns[5 + i * 4 + 1]
                props.append(float(prop))
            train_prop1_data.append(props)
            props = []
            for i in range(6):
                prop = columns[5 + i * 4 + 2]
                props.append(float(prop))
            train_prop2_data.append(props)
    return train_seq_data, train_prop1_data, train_prop2_data


# ======================================================
# make train data

file_path = '../data/ecoli_er2925.pcr_MSssI.r9.timp.061716.pass.fileter.50000.3250000.cpgfeatures.gama.tsv'

pos_train_seq_data, pos_train_prop1_data, pos_train_prop2_data = load_data(file_path)
print('positive data size:', len(pos_train_seq_data))

t = (pos_train_seq_data, pos_train_prop1_data, pos_train_prop2_data, [1] * len(pos_train_seq_data))
with open('../cache/train_pos_data.pkl', 'wb') as file:
    pickle.dump(t, file)

file_path = '../data/ecoli_er2925.pcr.r9.timp.061716.pass.filter.50000.3250000.cpgfeatures.gama.tsv'
neg_train_seq_data, neg_train_prop1_data, neg_train_prop2_data = load_data(file_path)

print('negative data size:', len(neg_train_seq_data))
t = (neg_train_seq_data, neg_train_prop1_data, neg_train_prop2_data, [0] * len(neg_train_seq_data))
with open('../cache/train_neg_data.pkl', 'wb') as file:
    pickle.dump(t, file)

# ======================================================
# make test data

file_path = '../data/NA12878.pcr_MSssI.r9.timp.081016.pass.nanopolish.cpgfeatures.gama.tsv'

pos_train_seq_data, pos_train_prop1_data, pos_train_prop2_data = load_data(file_path)
print('positive data size:', len(pos_train_seq_data))

t = (pos_train_seq_data, pos_train_prop1_data, pos_train_prop2_data, [1] * len(pos_train_seq_data))
with open('../cache/eval_pos_data.pkl', 'wb') as file:
    pickle.dump(t, file)

file_path = '../data/NA12878.pcr.r9.timp.081016.pass.nanopolish.cpgfeatures.gama.tsv'
neg_train_seq_data, neg_train_prop1_data, neg_train_prop2_data = load_data(file_path)

print('negative data size:', len(neg_train_seq_data))
t = (neg_train_seq_data, neg_train_prop1_data, neg_train_prop2_data, [0] * len(neg_train_seq_data))
with open('../cache/eval_neg_data.pkl', 'wb') as file:
    pickle.dump(t, file)
