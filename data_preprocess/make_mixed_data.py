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

total_lens = len(total_train_seq_data)
train_eval_line = int(train_eval_rate * total_lens)

with open('../cache/total_train_seq_data.pkl', 'wb') as file:
    pickle.dump(total_train_seq_data, file)
with open('../cache/total_prop1_data.pkl', 'wb') as file:
    pickle.dump(total_prop1_data, file)
with open('../cache/total_prop2_data.pkl', 'wb') as file:
    pickle.dump(total_prop2_data, file)
with open('../cache/total_label.pkl', 'wb') as file:
    pickle.dump(total_label, file)
