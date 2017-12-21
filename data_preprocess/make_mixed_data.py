# encoding=utf-8
import pickle
import numpy as np

train_eval_rate = 0.9

file_path_list = ['../cache/train_pos_data.pkl', '../cache/train_neg_data.pkl', '../cache/eval_pos_data.pkl',
                  '../cache/eval_neg_data.pkl']

# read all data
total_train_seq_data = []
total_prop1_data = []
total_prop2_data = []
total_label = []

for file_path in file_path_list:
    with open(file_path, 'rb') as file:
        train_seq_data, train_prop1_data, train_prop2_data, label = pickle.load(file)
        total_train_seq_data = total_train_seq_data + train_seq_data
        total_prop1_data = total_prop1_data + train_prop1_data
        total_prop2_data = total_prop2_data + train_prop2_data
        total_label = total_label + label

mark = list(range(len(total_label)))
np.random.shuffle(mark)
total_train_seq_data = np.array(total_train_seq_data)[mark]
total_prop1_data = np.array(total_prop1_data)[mark]
total_prop2_data = np.array(total_prop2_data)[mark]
total_label = np.array(total_label)[mark]

total_lens = len(total_train_seq_data)
train_eval_line = int(train_eval_rate * total_lens)

train_data = (
    total_train_seq_data[:train_eval_line], total_prop1_data[:train_eval_line], total_prop2_data[:train_eval_line],
    total_label[:train_eval_line])

eval_data = (
    total_train_seq_data[train_eval_line:], total_prop1_data[train_eval_line:], total_prop2_data[train_eval_line:],
    total_label[train_eval_line:])

with open('../cache/mixed_train_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)
with open('../cache/mixed_eval_data.pkl', 'wb') as file:
    pickle.dump(eval_data, file)
