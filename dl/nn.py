# @Author:      HgS_1217_
# @Create Date: 2018/6/19

import numpy as np
import tensorflow as tf

from config import CKPT_PATH, LOG_PATH, CKPT_PREFIX
from dl.utils import fc_layer, add_loss_summaries


class NN:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb=0.5, epoch_size=100,
                 learning_rate=0.001, loss_array=None, start_step=0, input_width=100, n_classes=2,
                 open_summary=False, new_ckpt_internal=0):
        self._raws = raws
        self._labels = labels
        self._test_raws = test_raws
        self._test_labels = test_labels
        self._keep_pb = keep_pb
        self._epoch_size = epoch_size
        self._start_step = start_step
        self._learning_rate = learning_rate
        self._loss_array = loss_array
        self._input_width = input_width
        self._classes = n_classes
        self._open_summary = open_summary
        self._new_ckpt_internal = new_ckpt_internal

        self._x = tf.placeholder(tf.float32, shape=[None, self._input_width],
                                 name="input_x")
        self._y = tf.placeholder(tf.float32, shape=[None, self._classes], name="input_y")
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._global_step = tf.Variable(0, trainable=False)

    def _build_network(self, x, y, is_training):
        flat = tf.reshape(x, [-1, 75])

        fc1 = fc_layer(flat, 256, is_training, "fc1")
        fc2 = fc_layer(fc1, 1024, is_training, "fc2")
        fc2_drop = tf.nn.dropout(fc2, self._keep_prob)
        fc3 = fc_layer(fc2_drop, 1024, is_training, "fc3")
        fc3_drop = tf.nn.dropout(fc3, self._keep_prob)
        fc4 = fc_layer(fc3_drop, 256, is_training, "fc4")
        fc5 = fc_layer(fc4, 1, is_training, "fc5", relu_flag=False)

        out = fc5
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs((y - out) / y), reduction_indices=[1]))
        accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32), axis=0)
        return loss, out, accu

    def _train_set(self, total_loss, global_step):
        loss_averages_op = add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self._learning_rate)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction, accu = self._build_network(self._x, self._y, self._is_training)
            train_op = self._train_set(loss, self._global_step)

            saver = tf.train.Saver()
            tf.add_to_collection('prediction', prediction)

            summary_op = tf.summary.merge_all()

            if self._start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            if self._open_summary:
                summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

            train_loss_pl = tf.placeholder(tf.float32)
            test_loss_pl = tf.placeholder(tf.float32)
            train_loss_summary = tf.summary.scalar("Train_average_loss_MAPE", train_loss_pl)
            test_loss_summary = tf.summary.scalar("Average_loss_MAPE", test_loss_pl)

            for step in range(self._start_step + 1, self._start_step + self._epoch_size + 1):
                print("Training epoch %d/%d" % (step, self._start_step + self._epoch_size))
                total_batch = len(self._raws)
                epoch_loss = np.zeros(total_batch)
                epoch_accu = np.zeros(total_batch)

                for bat in range(total_batch):
                    batch_xs = self._raws[bat]
                    batch_ys = self._labels[bat]
                    _, sum_str, pd, epoch_loss[bat], epoch_accu[bat] = sess.run(
                        [train_op, summary_op, prediction, loss, accu],
                        feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: self._keep_pb,
                                   self._is_training: True})

                print("Training epoch %d/%d finished, loss %g, accu %g" %
                      (step, self._start_step + self._epoch_size, np.mean(epoch_loss), np.mean(epoch_accu)))
                self._print_class_accu(np.mean(epoch_loss, axis=0), np.mean(epoch_accu, axis=0))
                print("==============================================================")

                if self._open_summary:
                    loss_str = sess.run([train_loss_summary], feed_dict={train_loss_pl: np.mean(epoch_loss)})
                    summary_writer.add_summary(loss_str, step)

                if step % 1 == 0:
                    print("Testing epoch %d/%d" % (step, self._start_step + self._epoch_size))
                    test_batch = len(self._test_raws)
                    test_loss = np.zeros(test_batch)
                    test_accu = np.zeros(test_batch)

                    for bat in range(test_batch):
                        batch_xs = self._test_raws[bat]
                        batch_ys = self._test_labels[bat]
                        pd, test_loss[bat], test_accu[bat] = sess.run(
                            [prediction, loss, accu],
                            feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: 1.0,
                                       self._is_training: False})

                    print("Testing epoch %d/%d finished, loss %g, accu %g" %
                          (step, self._start_step + self._epoch_size, np.mean(test_loss), np.mean(test_accu)))
                    self._print_class_accu(np.mean(test_loss, axis=0), np.mean(test_accu, axis=0))
                    print("==============================================================")

                    if self._open_summary:
                        test_loss_str, test_f1_str = sess.run([test_loss_summary],
                                                              feed_dict={test_loss_pl: np.mean(test_loss)})
                        summary_writer.add_summary(test_loss_str, step)

                print("saving model.....")
                if self._new_ckpt_internal == 0:
                    saver.save(sess, CKPT_PATH)
                elif self._new_ckpt_internal > 0:
                    path = "{0}{1}/model.ckpt".format(CKPT_PREFIX, int((step - 1) / self._new_ckpt_internal))
                    saver.save(sess, path)
                print("end saving....\n")