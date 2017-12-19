# encoding=utf-8
import tensorflow as tf
from dataloader import DataLoader
from sklearn import metrics
from sequence_model import SeqModel

batch_size = 512
show_step = 200

# ===================================

loader = DataLoader(train_mode=False)
init_learning_rate = 0.0005
decay_rate = 0.96
decay_steps = loader.train_size / batch_size

print('Test dataset size:', loader.train_size)

model = SeqModel(init_learning_rate, decay_steps, decay_rate)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, '../cache/sequence_model')
    sess.run(tf.local_variables_initializer())
    pred_collection = []
    logit_collection = []
    for step, index in enumerate(range(0, loader.train_size, batch_size)):
        batch_seq = loader.train_seq_data[index:index + batch_size]
        batch_prob1 = loader.train_prop1_data[index:index + batch_size]
        batch_prob2 = loader.train_prop2_data[index:index + batch_size]
        batch_label = loader.train_label[index:index + batch_size]

        # auc op is updated at each step
        y_pred, logits, auc = sess.run([model.prediction, model.activation_logits, model.auc_opt],
                                       feed_dict={model.seq_data: batch_seq, model.prob1_data: batch_prob1,
                                                  model.prob2_data: batch_prob2, model.label: batch_label,
                                                  model.keep_prob: 1.0})

        pred_collection.extend(y_pred)
        logit_collection.extend(logits)
        print('%d -- %d' % (step + 1, decay_steps))

    print("accuracy: %.3f, auc: %.6f" % (
        metrics.accuracy_score(loader.train_label, pred_collection),
        metrics.roc_auc_score(loader.train_label, logit_collection)))
    print("Precision %.6f" % metrics.precision_score(loader.train_label, pred_collection))
    print("Recall %.6f" % metrics.recall_score(loader.train_label, pred_collection))
    print("F1-score %.6f" % metrics.f1_score(loader.train_label, pred_collection))

    print('auc calculated in tensorflow:', auc)
