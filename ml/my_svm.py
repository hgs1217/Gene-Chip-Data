# -*- coding: utf-8 -*-  
"""
@author: Suibin Sun
@file: svm.py
@time: 2018/6/28 12:10
"""

import numpy as np
import tensorflow as tf
from config import CKPT_PATH


class MY_SVM:
    def __init__(self, raw_ids, raw_data, raw_labels, batch_size=None, input_dim=22283, alpha=0.01, learning_rate=0.01,
                 epoch_size=1000, start_step=0):
        """
        有点小问题，@hgs你有时间的话可以帮忙看下哪里写错了
        :param raw_ids:
        :param raw_data:
        :param raw_labels:
        :param batch_size:
        :param input_dim:
        :param alpha:
        :param learning_rate:
        :param epoch_size:
        :param start_step:
        """
        self.__ids = raw_ids
        self.__data = raw_data
        self.__labels = raw_labels
        self.__X = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
        self.__y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.__batch_size = batch_size
        self.__A = tf.Variable(tf.random_normal(shape=[input_dim, 1]))
        self.__b = tf.Variable(tf.random_normal(shape=[1, 1]))
        self.__alpha = alpha
        self.__learning_rate = learning_rate
        self.__epoch_size = epoch_size
        self.__start_step = start_step

    def split(self):
        train_ids, train_data, train_labels, test_ids, test_data, test_labels = [], [], [], [], [], []
        for i in range(len(self.__ids)):
            if i % 6 == 5:
                test_ids.append(self.__ids[i])
                test_data.append(self.__data[i])
                test_labels.append(self.__labels[i])
            else:
                train_ids.append(self.__ids[i])
                train_data.append(self.__data[i])
                train_labels.append(self.__labels[i])
        return (train_ids, train_data, train_labels), (test_ids, test_data, test_labels)

    def evaluate(self, out, y):
        pred_out = tf.maximum(0., tf.sign(out))
        accu = tf.metrics.accuracy(y, pred_out)
        pre = tf.metrics.precision(y, pred_out)
        recall = tf.metrics.recall(y, pred_out)
        f1 = tf.divide(tf.multiply(2., tf.multiply(pre, recall)), tf.add(pre, recall))

        l1_norm = tf.reduce_sum(tf.abs(self.__A))
        # l2_norm = tf.reduce_sum(tf.square(self.__A))
        alpha = tf.constant([self.__alpha])

        hinge_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(out, y))))

        loss = tf.add(hinge_loss, tf.multiply(alpha, l1_norm))
        return loss, accu, pre, recall, f1

    def model(self):
        model_output = tf.subtract(tf.matmul(self.__X, self.__A), self.__b)

        loss, accu, pre, recall, f1 = self.evaluate(model_output, self.__y)

        return model_output, loss, accu, pre, recall, f1

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        (train_ids, train_data, train_labels), (test_ids, test_data, test_labels) = self.split()

        with tf.Session(config=config) as sess:
            out, loss, accu, pre, recall, f1 = self.model()
            opt = tf.train.GradientDescentOptimizer(self.__learning_rate)
            train_step = opt.minimize(loss)
            saver = tf.train.Saver()

            sess.run(tf.local_variables_initializer())
            if self.__start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            for step in range(self.__start_step + 1, self.__start_step + self.__epoch_size + 1):
                _, _loss, _accu, _f1 = sess.run([train_step, loss, accu, f1],
                                                feed_dict={self.__X: train_data, self.__y: train_labels})
                print(f'Training epoch {step}/{self.__start_step + self.__epoch_size} finished, '
                      f'loss={_loss}, accu={_accu}, f1={_f1}')
                saver.save(sess, CKPT_PATH)

                if step % 10 == 0:
                    # sess.run(tf.local_variables_initializer())
                    pre_loss, pre_accu, pre_f1 = sess.run([loss, accu, f1],
                                                          feed_dict={self.__X: test_data, self.__y: test_labels})
                    print(f'\tTesting epoch {step}/{self.__start_step + self.__epoch_size} finished, '
                          f'loss={pre_loss}, accu={pre_accu}, f1={pre_f1}')
