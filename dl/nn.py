# @Author:      HgS_1217_
# @Create Date: 2018/6/19

import numpy as np
import tensorflow as tf

from config import CKPT_PATH, LOG_PATH, CKPT_PREFIX
from dl.utils import fc_layer, add_loss_summaries, weighted_loss


class NN:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb=0.5, epoch_size=100,
                 learning_rate=0.001, loss_array=None, start_step=0, input_width=22283, n_classes=2,
                 new_ckpt_internal=0):
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
        self._new_ckpt_internal = new_ckpt_internal

        self._x = tf.placeholder(tf.float32, shape=[None, self._input_width], name="input_x")
        self._y = tf.placeholder(tf.float32, shape=[None, self._classes], name="input_y")
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._global_step = tf.Variable(0, trainable=False)

    def _build_network(self, x, y, is_training):
        fc1 = fc_layer(x, 256, is_training, "fc1")
        fc2 = fc_layer(fc1, 1024, is_training, "fc2")
        fc2_drop = tf.nn.dropout(fc2, self._keep_prob)
        fc3 = fc_layer(fc2_drop, 1024, is_training, "fc3")
        fc3_drop = tf.nn.dropout(fc3, self._keep_prob)
        fc4 = fc_layer(fc3_drop, 256, is_training, "fc4")
        fc5 = fc_layer(fc4, self._classes, is_training, "fc5", relu_flag=False)
        norm5 = tf.nn.softmax(tf.nn.sigmoid(fc5))

        out = norm5
        loss = weighted_loss(out, y, self._loss_array)
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

    def _print_class_accu(self, loss, accu):
        for i in range(self._classes):
            print("\tclass %d, loss %g, accu %g" % (i, loss[i], accu))

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction, accu = self._build_network(self._x, self._y, self._is_training)
            train_op = self._train_set(loss, self._global_step)

            saver = tf.train.Saver()
            tf.add_to_collection('prediction', prediction)

            if self._start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            for step in range(self._start_step + 1, self._start_step + self._epoch_size + 1):
                print("Training epoch %d/%d" % (step, self._start_step + self._epoch_size))

                _, pd, epoch_loss, epoch_accu = sess.run(
                    [train_op, prediction, loss, accu],
                    feed_dict={self._x: np.array(self._raws),
                               self._y: np.array(self._labels),
                               self._keep_prob: self._keep_pb,
                               self._is_training: True})

                print("Training epoch %d/%d finished, loss %g, accu %g" %
                      (step, self._start_step + self._epoch_size, np.mean(epoch_loss), epoch_accu))
                self._print_class_accu(epoch_loss, epoch_accu)
                print("==============================================================")

                if step % 1 == 0:
                    print("Testing epoch %d/%d" % (step, self._start_step + self._epoch_size))

                    pd, test_loss, test_accu = sess.run(
                        [prediction, loss, accu],
                        feed_dict={self._x: np.array(self._test_raws),
                                   self._y: np.array(self._test_labels),
                                   self._keep_prob: 1.0,
                                   self._is_training: False})

                    print("Testing epoch %d/%d finished, loss %g, accu %g" %
                          (step, self._start_step + self._epoch_size, np.mean(test_loss), test_accu))
                    self._print_class_accu(test_loss, test_accu)
                    print("==============================================================")

                print("saving model.....")
                if self._new_ckpt_internal == 0:
                    saver.save(sess, CKPT_PATH)
                elif self._new_ckpt_internal > 0:
                    path = "{0}{1}/model.ckpt".format(CKPT_PREFIX, int((step - 1) / self._new_ckpt_internal))
                    saver.save(sess, path)
                print("end saving....\n")