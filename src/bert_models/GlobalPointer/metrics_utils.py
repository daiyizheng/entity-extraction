#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 23:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metrics_utils.py
import numpy as np

class MetricsCalculator(object):
    def __init__(self):
        pass

    def metric_result(self, y_pred, y_true):
        """
        [bz, num_label, max_len, max_len]
        :param y_pred:
        :param y_true:
        :return:
        """
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        try:
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        except:
            f1, precision, recall = 0, 0, 0
        return f1, precision, recall

