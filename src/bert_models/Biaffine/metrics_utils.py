#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 13:37
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : finetuning_argparse.py

import csv
import json
import torch
import time
import logging

# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report  # 专门用于序列标注评分用的
import numpy as np

logger = logging.getLogger(__file__)

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "acc":accuracy_score(labels, preds),
        "frame_acc":get_sentence_frame_acc(preds, labels)["sementic_frame_acc"]
    }


def get_sentence_frame_acc(slot_preds, slot_labels):
    """For the cases that all the slots are correct (in one sentence)"""

    # Get the slot comparision result
    slot_result = []    # 一整句话全部预测正确的就为True
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = slot_result.mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(slot_preds, slot_labels)

    results.update(slot_result)
    results.update(sementic_result)

    return results


# def compute_metrics_sklearn(intent_preds, intent_labels):
#     """
#     evaluation metrics
#     :param intent_preds: prediction labels
#     :param intent_labels:glod labels
#     :return:dict {}
#     """
#     assert len(intent_preds) == len(intent_labels)
#     results = {}
#     classification_report_dict = classification_report(intent_labels, intent_preds, output_dict=True)
#     auc = roc_auc_score(intent_labels, intent_preds)
#     results["auc"] = auc
#     for key0, val0 in classification_report_dict.items():
#         if isinstance(val0, dict):
#             for key1, val1 in val0.items():
#                 results[key0 + "__" + key1] = val1
#
#         else:
#             results[key0] = val0
#     return results

def model_metrics(true_labels, pre_labels):
    """
    :param true_labels 真实标签数据 [O,O,B-OR, I-OR]
    :param pre_labels 预测标签数据 [O,O,B-OR, I-OR]
    :param logger 日志实例
    """
    start_time = time.time()
    acc = accuracy_score(true_labels, pre_labels)
    f1score = f1_score(true_labels, pre_labels, average='macro')
    report = classification_report(true_labels, pre_labels, digits=4)
    precision = precision_score(true_labels, pre_labels)
    recall = recall_score(true_labels, pre_labels)
    msg = '\nTest Acc: {0:>6.2%}, Test f1: {1:>6.2%}'
    logger.info(msg.format(acc, f1score))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info("\n{}".format(report))
    time_dif = time.time() - start_time
    logger.info("Time usage:{0:>.6}s".format(time_dif))
    return {"precision":precision, "recall":recall, "f1":f1score, "acc":acc}

