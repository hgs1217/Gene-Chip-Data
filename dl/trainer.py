# @Author:      HgS_1217_
# @Create Date: 2018/6/24

import os

from dl.nn import NN
from preprocess.data_processor import read_data


def train(start_step=0, epoch_size=100, keep_pb=0.5, learning_rate=0.001,
          open_summary=False, new_ckpt_internal=0, gpu=True):
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use cpu only

    ids, data = read_data()
    raws, test_raws = data[:-3000], data[-3000:]

    nn = NN(raws, labels, test_raws, test_labels, epoch_size=epoch_size, loss_array=loss_array,
              start_step=start_step, keep_pb=keep_pb, learning_rate=learning_rate,
              open_summary=open_summary, new_ckpt_internal=new_ckpt_internal)
    nn.train()
