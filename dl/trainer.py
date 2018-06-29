# @Author:      HgS_1217_
# @Create Date: 2018/6/24

import os
import numpy as np

from dl.nn import NN
from preprocess.data_processor import read_data, read_srdf


def split(ids, data, labels):
    train_ids, train_data, train_labels, test_ids, test_data, test_labels = [], [], [], [], [], []
    cnt = 0
    for i in range(len(ids)):
        if labels[i][0] == 1:
            cnt += 1
        if i % 6 == 5:
            test_ids.append(ids[i])
            test_data.append(data[i])
            test_labels.append(np.array([0, 1]) if labels[i][0] == 1 else np.array([1, 0]))
        else:
            train_ids.append(ids[i])
            train_data.append(data[i])
            train_labels.append(np.array([0, 1]) if labels[i][0] == 1 else np.array([1, 0]))
    loss_array = [0.5 / (cnt / len(labels)), 0.5 / (1 - cnt / len(labels))]
    return (train_ids, train_data, train_labels), (test_ids, test_data, test_labels), loss_array


def train(start_step=0, epoch_size=100, keep_pb=0.5, learning_rate=0.001,
          new_ckpt_internal=0, gpu=True):
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use cpu only

    ids, data = read_data()
    _, labels = read_srdf()
    print(labels)
    (_, train_data, train_labels), (_, test_data, test_labels), loss_array = split(ids, data, labels)

    nn = NN(train_data, train_labels, test_data, test_labels, epoch_size=epoch_size, loss_array=loss_array,
            start_step=start_step, keep_pb=keep_pb, learning_rate=learning_rate,
            new_ckpt_internal=new_ckpt_internal)
    nn.train()
