#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 22:53
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : loss_function_utils.py.py
from __future__ import annotations


import torch
import torch.nn as nn

class multilabel_categorical_crossentropy(nn.Module):
    def __init__(self):
        super(multilabel_categorical_crossentropy, self).__init__()

    # 损失函数公式 log(1+∑e^{s_i}) +log(1+∑e^{-s_j}) (i \in neg, j \in pos)
    def forward(self, y_pred, y_true):
        # 负例: (1 - 2 * 0) * y_pred = y_pred  正例: (1 - 2 * 1) * y_pred = - y_pred
        y_pred = (1 - 2 * y_true) * y_pred  # 预测为1， 真实 1， 0
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes，
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # e^0=1
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()