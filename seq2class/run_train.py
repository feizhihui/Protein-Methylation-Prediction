# encoding=utf-8
import tensorflow as tf
from dataloader import DataLoader
from sklearn import metrics
from sequence_model import SeqModel

batch_size = 512
epoch_num = 16
show_step = 200
keep_prob = 0.75
# ===================================

loader = DataLoader()

# learning rate exponential decay
init_learning_rate = 0.001
decay_rate = 0.96
decay_steps = loader.train_size / batch_size

print('Train dataset size:', loader.train_size)
model = SeqModel(init_learning_rate, decay_steps, decay_rate)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epoch_num):
        print('========== epoch - ', str(epoch + 1), "===================")
        loader.shuffle()
        for step, index in enumerate(range(0, loader.train_size, batch_size)):
            batch_seq = loader.train_seq_data[index:index + batch_size]
            batch_prob1 = loader.train_prop1_data[index:index + batch_size]
            batch_prob2 = loader.train_prop2_data[index:index + batch_size]
            batch_label = loader.train_label[index:index + batch_size]

            sess.run(model.train_op,
                     feed_dict={model.seq_data: batch_seq, model.prob1_data: batch_prob1, model.prob2_data: batch_prob2,
                                model.label: batch_label, model.keep_prob: keep_prob})
            if step % show_step == 0:
                y_pred, batch_cost, batch_accuracy, auc = sess.run(
                    [model.prediction, model.cost, model.accuracy, model.auc_opt], feed_dict={model.seq_data: batch_seq,
                                                                                              model.prob1_data: batch_prob1,
                                                                                              model.prob2_data: batch_prob2,
                                                                                              model.label: batch_label,
                                                                                              model.keep_prob: 1.})
                print("cost function: %.3f, accuracy: %.3f, auc: %.3f" % (batch_cost, batch_accuracy, auc))
                print("Precision %.6f" % metrics.precision_score(batch_label, y_pred))
                print("Recall %.6f" % metrics.recall_score(batch_label, y_pred))
                print("F1-score %.6f" % metrics.f1_score(batch_label, y_pred))
                print('Accuracy:%.3f' % metrics.accuracy_score(batch_label, y_pred))
                print('step %d -- total step %d -- epoch %d' % (step + 1, decay_steps, epoch + 1))

    # store
    saver = tf.train.Saver()
    saver.save(sess, '../cache/sequence_model')
