# -*- coding: utf-8 -*-  
"""
@author: Suibin Sun
@file: lll.py
@time: 2018/6/28 10:20
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import json

dic = {}

with open('D:/All_Projects/DataMining_Project/Gene_Chip_Data/labels.json', 'r') as f:
    for line in f.readlines():
        line = line.split('\t')
        dic[line[0].strip()] = int(line[1].strip())
    with open('labels.json', 'w') as ff:
        json.dump(dic, ff, indent=2)
