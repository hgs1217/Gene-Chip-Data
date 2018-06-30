# -*- coding: utf-8 -*-  
"""
@author: Suibin Sun
@file: logistic_regression.py
@time: 2018/6/28 23:37
"""

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pickle
from config import CKPT_PREFIX
from ml.plot import plot_learning_curve, plot_roc
import matplotlib.pyplot as plt


class SK_LR:
    def __init__(self, raw_ids, raw_data, raw_labels, train=True, pk_name='sk_lr', pca=False, n_components=2000):
        self.__ids = raw_ids
        self.__data = raw_data
        self.__labels = [i[0] for i in raw_labels]
        self.__clf = LogisticRegression(penalty='l1', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                                        intercept_scaling=1, class_weight='balanced', random_state=None, solver='saga',
                                        max_iter=1000, multi_class='ovr', verbose=1, warm_start=True, n_jobs=-1)
        # self.__clf = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
        #                                 intercept_scaling=1, class_weight='balanced', random_state=None, solver='sag',
        #                                 max_iter=1000, multi_class='ovr', verbose=1, warm_start=True, n_jobs=-1)
        self.__train = train
        self.__pk_name = pk_name
        self.__pca = pca
        self.__n_components = n_components

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

    def train(self):
        (train_ids, train_data, train_labels), (test_ids, test_data, test_labels) = self.split()
        if self.__pca:
            original_dim = len(train_data[0])
            pca = PCA(n_components=self.__n_components, copy=False)
            all_data = train_data + test_data
            pca.fit(all_data)
            train_data = pca.transform(train_data)
            test_data = pca.transform(test_data)
            print(f'pca finished, dim {original_dim} -> {len(train_data[0])}')
            print(np.array(train_data).shape, np.array(test_data).shape)

        if self.__train:
            self.__clf.fit(train_data, train_labels)
            with open(CKPT_PREFIX + self.__pk_name + '.pk', 'wb') as f:
                pickle.dump(self.__clf, f)
        else:
            with open(CKPT_PREFIX + self.__pk_name + '.pk', 'rb') as f:
                self.__clf = pickle.load(f)
                # plot_learning_curve(self.__clf, 'Learning Curves (Logistic Regression)', self.__data, self.__labels,
                #                     cv=6)
                # plt.savefig('logistic.jpg')
                # plt.show()
                y_score = self.__clf.predict_proba(test_data)
                y_score = np.array(y_score)[:, 1]
                plot_roc(test_labels, y_score, 'LR','')
        print('\tLR')
        pred_labels = self.__clf.predict(test_data)
        print(f'Accuracy: {metrics.accuracy_score(test_labels,pred_labels)}')
        print(f'Precision: {metrics.precision_score(test_labels,pred_labels)}')
        print(f'Recall: {metrics.recall_score(test_labels,pred_labels)}')
        print(f'f1 score: {metrics.f1_score(test_labels,pred_labels)}')
