# encoding=utf-8
import numpy as np
import pickle


class DataLoader(object):
    def __init__(self, train_mode=True):
        if train_mode:
            file_path = '../cache/train_pos_data.pkl'

        else:
            file_path = '../cache/eval_pos_data.pkl'


        with open(pos_file_path, 'rb') as file:
            pos_train_seq_data, pos_train_prop1_data, pos_train_prop2_data, pos_label = pickle.load(file)
        with open(neg_file_path, 'rb') as file:
            neg_train_seq_data, neg_train_prop1_data, neg_train_prop2_data, neg_label = pickle.load(file)

        self.train_seq_data = np.array(pos_train_seq_data + neg_train_seq_data, dtype=np.int32)
        self.train_prop1_data = np.array(pos_train_prop1_data + neg_train_prop1_data,
                                         dtype=np.float32)
        self.train_prop2_data = np.array(pos_train_prop2_data + neg_train_prop2_data,
                                         dtype=np.float32)
        self.train_label = np.array(pos_label + neg_label, np.int32)
        self.train_size = len(self.train_label)

        # self.MaxMinNormalization(train_mode)
        self.ZScoreNormalization(train_mode)

    def shuffle(self):
        mark = list(range(self.train_size))
        np.random.shuffle(mark)
        self.train_seq_data = self.train_seq_data[mark]
        self.train_prop1_data = self.train_prop1_data[mark]
        self.train_prop2_data = self.train_prop2_data[mark]
        self.train_label = self.train_label[mark]

    def MaxMinNormalization(self, train_mode):
        if train_mode:
            with open('../cache/maxmin_prob.txt', 'w') as file:
                pass
                min_v = np.min(self.train_prop1_data)
                max_v = np.max(self.train_prop1_data)
                self.train_prop1_data = (self.train_prop1_data - min_v) / (max_v - min_v)
                file.write(str(min_v) + " " + str(max_v) + "\n")
                min_v = np.min(self.train_prop2_data)
                max_v = np.max(self.train_prop2_data)
                self.train_prop2_data = (self.train_prop2_data - min_v) / (max_v - min_v)
                file.write(str(min_v) + " " + str(max_v) + "\n")
        else:
            with open('../cache/maxmin_prob.txt', 'r') as file:
                s = file.readline().split()
                min_v, max_v = float(s[0]), float(s[1])
                self.train_prop1_data = (self.train_prop1_data - min_v) / (max_v - min_v)
                s = file.readline().split()
                min_v, max_v = float(s[0]), float(s[1])
                self.train_prop2_data = (self.train_prop2_data - min_v) / (max_v - min_v)

    def ZScoreNormalization(self, train_mode):
        if train_mode:
            with open('../cache/zscore_prob.txt', 'w') as file:
                mu = np.mean(self.train_prop1_data)
                sigma = np.std(self.train_prop1_data)
                self.train_prop1_data = (self.train_prop1_data - mu) / sigma
                file.write(str(mu) + " " + str(sigma) + "\n")
                mu = np.mean(self.train_prop2_data)
                sigma = np.std(self.train_prop2_data)
                self.train_prop2_data = (self.train_prop2_data - mu) / sigma
                file.write(str(mu) + " " + str(sigma) + "\n")
        else:
            with open('../cache/zscore_prob.txt', 'r') as file:
                s = file.readline().split()
                mu, sigma = float(s[0]), float(s[1])
                self.train_prop1_data = (self.train_prop1_data - mu) / sigma
                s = file.readline().split()
                mu, sigma = float(s[0]), float(s[1])
                self.train_prop2_data = (self.train_prop2_data - mu) / sigma


if __name__ == '__main__':
    DataLoader()
