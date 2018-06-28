# @Author:      HgS_1217_
# @Create Date: 2018/6/24
import json

import numpy as np
from config import DATASET_PATH, DATA_FILE_PATH, DATA_SDRF_PATH, READ_LB_PATH


def read_data(data_file_path=DATA_FILE_PATH):
    """
    :param data_file_path:
    :return: ids -- list, shape (5896,), like ['Hyb_1', 'Hyb_2', ...]
            data -- np array, shape (5896, 22283)
    """
    with open(data_file_path, "r") as f:
        id_line = f.readline()
        ids = [x.strip()[13:-2] for x in id_line.split('\t')[1:]]  # 取第一行的ID，切去无意义的字段，最终格式："Hyb_X"
        data = np.zeros((5896, 22283), dtype=np.float32)  # 形状 [5896, 22283]
        for i in range(22283):
            line = f.readline()
            if i % 1000 == 0:
                print("Reading line %d" % i)
            line = line.split("\t")
            data[:, i] = np.array(list(map(lambda x: float(x), line[1:])))

    print("Read data complete")
    print(data.shape)
    return ids, data


def read_srdf(data_file_path=DATA_SDRF_PATH):
    """
    :param data_file_path:
    :return: ids -- list, shape (5896,), like ['Hyb_1', 'Hyb_2', ...]
            labels -- list, shape (5896, 1) like [[1], [0], ..., [1]]
    """
    with open(data_file_path, "r") as f:
        with open(READ_LB_PATH, 'r') as fl:
            label_dict = json.load(fl)      # 得到文本->0或1的dict
            f.readline()
            ids = []
            labels = []
            for i in range(5896):
                line = f.readline()
                line = line.split("\t")
                ids.append(line[14])            # 第14列的ID
                label_text = line[7].strip()    # 取第7列的label文本
                if len(label_text) > 0:     # 不是空字段
                    # 去掉双引号
                    if label_text[0] == '"': label_text = label_text[1:]
                    if label_text[-1] == '"': label_text = label_text[:-1]
                labels.append([label_dict[label_text]])

    print("Read labels complete")
    print(np.array(labels).shape)
    return ids, labels


if __name__ == '__main__':
    (ids, data), (ids2, labels) = (read_data()), (read_srdf())
    for i in range(10):
        print(ids[i], data[i], ids2[i], labels[i])
