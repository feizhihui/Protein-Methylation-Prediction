# encoding=utf-8
import tensorflow as tf
import numpy as np

n_hidden = 200

sequence_lens = 6
threshold = 0.5

vocab_size = 1965
embedding_size = 128


class SeqModel(object):
    def __init__(self, init_learning_rate, decay_steps, decay_rate):
        self.seq_data = tf.placeholder(tf.int32, [None, sequence_lens])
        self.prob1_data = tf.placeholder(tf.float32, [None, sequence_lens])
        self.prob2_data = tf.placeholder(tf.float32, [None, sequence_lens])
        self.label = tf.placeholder(tf.int32, [None])

        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate,
                                                   staircase=True)

        # input_x = tf.one_hot(self.seq_data, depth=vocab_size, dtype=tf.int32)
        W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=np.sqrt(2. / vocab_size)), name="W",
                        dtype=tf.float32)
        embedded_x = tf.nn.embedding_lookup(W, self.seq_data)

        fusion_vector = tf.concat(
            [embedded_x, tf.reshape(self.prob1_data, [-1, sequence_lens, 1]),
             tf.reshape(self.prob2_data, [-1, sequence_lens, 1])], axis=2)

        # Current data input shape: (batch_size, n_steps, n_input)
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.keep_prob)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.keep_prob)
        # network = rnn_cell.MultiRNNCell([lstm_fw_cell, lstm_bw_cell] * 3)
        # x shape is [batch_size, max_time, input_size]
        outputs, output_sate = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, fusion_vector,
                                                               sequence_length=tf.ones_like(self.label,
                                                                                            dtype=tf.int32) * sequence_lens,
                                                               dtype=tf.float32)  # tf bug

        # # outputs, output_sate = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32)
        # # shape is n*40*(n_hidden+n_hidden) because of forward + backward
        outputs = (outputs[0][:, -1, :], outputs[1][:, 0, :])
        outputs = tf.concat(outputs, 1)

        with tf.name_scope("sigmoid_layer"):
            weights = tf.Variable(tf.truncated_normal([2 * n_hidden, 1]) * np.sqrt(2.0 / (2 * n_hidden)),
                                  dtype=tf.float32)
            bias = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)
            logits = tf.matmul(outputs, weights) + bias
            self.activation_logits = tf.nn.sigmoid(logits)

        with tf.name_scope("evaluation"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32),
                                                           logits=tf.squeeze(logits, axis=1))
            self.cost = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost,
                                                                                         global_step=global_step)
            # self.prediction = tf.cast(tf.greater_equal(self.activation_logits, threshold), tf.int32)
            self.prediction = tf.where(tf.greater(self.activation_logits, threshold),
                                       tf.ones_like(self.activation_logits, dtype=tf.int32),
                                       tf.zeros_like(self.activation_logits, dtype=tf.int32))

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.squeeze(self.prediction, axis=1), self.label),
                        tf.float32))  # dimension must be equal in equal
            self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.activation_logits, self.label)
