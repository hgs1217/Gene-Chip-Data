# @Author:      HgS_1217_
# @Create Date: 2018/6/19

import numpy as np
import tensorflow as tf


def variable_with_weight_decay(name, shape, initializer, wd=None):
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op


def batch_norm_layer(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = variable_with_weight_decay('beta', params_shape, initializer=tf.truncated_normal_initializer())
    gamma = variable_with_weight_decay('gamma', params_shape, initializer=tf.truncated_normal_initializer())

    batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv2d(x, w, stride, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, ksize, stride, feature_num, is_training, name=None, padding="SAME", groups=1):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [ksize, ksize, int(x.get_shape()[-1]) / groups, feature_num], dtype="float")
        b = tf.get_variable("b", [feature_num], dtype="float")

        x_split = tf.split(x, groups, 3)
        w_split = tf.split(w, groups, 3)

        feature_map_list = [conv2d(x_, w_, stride, padding) for x_, w_ in zip(x_split, w_split)]
        feature_map = tf.concat(feature_map_list, 3)

        out = tf.nn.bias_add(feature_map, b)
        norm = batch_norm_layer(out, is_training)
        feature_shape = list(map(lambda x: -1 if not x else x, feature_map.get_shape().as_list()))
        return tf.nn.relu(tf.reshape(norm, feature_shape), name=scope.name)


def avg_pool_layer(x, ksize, stride, name=None, padding="SAME"):
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_layer(x, feature_num, is_training, name=None, relu_flag=True):
    with tf.variable_scope(name) as scope:
        w = variable_with_weight_decay('w', shape=[x.get_shape()[-1], feature_num],
                                       initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [feature_num], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(tf.matmul(x, w), b)
        norm = batch_norm_layer(bias, is_training)
        return tf.nn.relu(norm) if relu_flag else norm


def weighted_loss(lgts, lbs, loss_array):
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-10)
        labels = lbs
        logits = lgts + epsilon

        cross_entropy = -tf.reduce_mean(tf.multiply(labels * tf.log(tf.nn.softmax(logits) + epsilon),
                                                   np.array(loss_array)), name='cross_entropy', axis=[0])
        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy', axis=0)
        return cross_entropy


def norm_layer(x, lsize, bias=1.0, alpha=0.001 / 9, beta=0.75):
    return tf.nn.lrn(x, lsize, bias=bias, alpha=alpha, beta=beta)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size, num_class = predictions.shape[0], predictions.shape[1]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].argmax(0).flatten(), predictions[i].argmax(0).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    acc = [0 for _ in range(num_class)]
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc[ii] = 0.0
        else:
            acc[ii] = np.diag(hist)[ii] / float(hist.sum(1)[ii])
    return np.nanmean(acc_total), np.nanmean(iu), acc
