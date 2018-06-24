# @Author:      HgS_1217_
# @Create Date: 2018/6/24

import numpy as np

from config import DATASET_PATH, DATA_FILE_PATH, DATA_SDRF_PATH


def read_data(data_file_path=DATA_FILE_PATH):
    with open(data_file_path, "r") as f:
        f.readline()
        ids = []
        data = np.zeros((22283, 5896), dtype=np.float32)
        for i in range(22283):
            line = f.readline()
            if i % 1000 == 0:
                print("Reading line %d" % i)
            line = line.split("\t")
            ids.append(line[0])
            data[i, :] = np.array(list(map(lambda x: float(x), line[1:])))

    print("Read data complete")
    print(data.shape)
    return ids, data


def read_labels(data_file_path=DATA_SDRF_PATH):
    with open(data_file_path, "r") as f:
        f.readline()
        ids = []
        labels = []
        for i in range(5896):
            line = f.readline()
            line = line.split("\t")
            ids.append(line[0])
            labels.append(line[1:11])

    print("Read labels complete")
    return ids, labels
