#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 13:36
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : utils.py
from typing import Optional, Text, Any, Dict, List, Tuple
import random
import os
import logging
import copy

import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def  init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    :param log_file: str, 文件目录
    :param log_file_level: str, 日志等级
    """
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file,encoding="utf-8")
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger

def get_labels(label_path):
    """
    read labels from file
    :param label_file: file path
    :return:
    """
    return [label.strip() for label in open(label_path, 'r', encoding='utf-8')]

def override_defaults(
    defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Override default config with the given config.
    code  from rasa 2.4.1
    We cannot use `dict.update` method because configs contain nested dicts.

    Args:
        defaults: default config
        custom: user config containing new parameters

    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            setattr(config, key, custom[key])

    return config


def conll_subword_label_alignment(token_list, label_list):
    text = ""
    entities = []
    index = 0
    while index<len(token_list):
        token = token_list[index]
        text = text + " " + token if text else token
        if label_list[index][0] == "B":
            label_type = label_list[index].split("-")[-1]
            start_ids = len(text) - len(token)
            metion = token
            end_ids = len(text)
            while(index+1<len(token_list) and label_list[index+1][0]=="I"):
                index += 1
                token = token_list[index]
                text = text + " " + token if text else token
                metion = metion+ " " + token
                end_ids = len(text)
            index += 1
            entities.append([start_ids, end_ids, metion, label_type])
        else:
            index += 1
    return text, entities


def subword_label_alignment(text, entities, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    label = ["O"] * len(offset_mapping)

    for loc in entities:  # {'id': 1, 'start': 0, 'end': 2, 'mention': 'EU', 'type': 'ORG'}
        start_char = int(loc["start"])
        end_char = int(loc["end"])
        entity_type = loc["type"]
        #token start index
        token_start_index = 0
        #token end index
        token_end_index = len(offset_mapping)-1
        while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
            token_start_index += 1
        token_start_index -= 1

        while offset_mapping[token_end_index][1] >= end_char and token_end_index!=-1:
            token_end_index -=1
        token_end_index += 1
        label[token_start_index] = "B-" + entity_type
        for idx in range(token_start_index+1, token_end_index+1):
            label[idx] = "I-" + entity_type
    return text, encoded["input_ids"], encoded["offset_mapping"], label

# def sub_token_label_scheme(token_list:List,
#                            label_list:List, tokenizer:PreTrainedTokenizerBase,
#                            scheme_type:Text="v1"):
#     if scheme_type=="v1":
#         text, entities = conll_subword_label_alignment(token_list, label_list)
#         encoded, labels = subword_label_alignment(text, entities, tokenizer)
#     else:
#         raise NotImplementedError
#     return text, encoded["input_ids"], encoded["offset_mapping"], labels


def label_list_to_entities(labels_list:List)->List[Tuple]:
    """

    :param labels_list:
    :return: [(start_ids, end_ids, type)]
    """
    entities = []
    entitiy = []

    for idx, label in enumerate(labels_list):
        # 开始B
        if label[0]=="B" and len(entitiy)==0:
            entitiy.append(idx)
            entitiy.append(0)
            entitiy.append(label.split("-")[1])
        elif label[0]=="B" and len(entitiy)!=0:
            entitiy[1] = (idx - 1)
            entities.append(tuple(entitiy))
            entitiy = [idx, 0, label.split("-")[1]]
        elif label[0]=="I":
            entitiy[1] = idx
        else:
            if entitiy:
                entitiy[1] = idx-1
                entities.append(tuple(entitiy))
            entitiy = []
    if entitiy:
        entitiy[1] = len(labels_list)-1
        entities.append(tuple(entitiy))

    return entities


def trans_label_MRC_EN(label_seq, matrix_lenth, label_map):  # [[8, 9, 'GPE'], [1, 6, 'ORG']]
    import numpy as np
    from scipy.sparse import coo_matrix
    import torch

    # conll 2 doccano json
    tmp_tokens = ["ant"] * matrix_lenth
    # print('tmp_tokens：', tmp_tokens)
    # print('label_seq:', label_seq)
    # _, list_spans = conll2doccano_json(tmp_tokens, label_seq)  # [[0, 1, '胸', '部位'], [2, 4, '腹部', '部位']]
    # print(list_spans)

    list_spans = [(span[0], span[1], span[2]) for span in label_seq]
    # print(list_spans)

    tmp_label = np.zeros((matrix_lenth, matrix_lenth))
    # print("tmp_label: ", tmp_label)

    for span in list_spans:
        tmp_label[span[0], span[1]] = label_map[span[2]]

    # print("tmp_label: ", tmp_label)

    label_sparse = coo_matrix(tmp_label, dtype=np.int)
    # print("label_sparse: ", label_sparse)

    values = label_sparse.data
    # print("values: ", values)

    # print("label_sparse.row: ", label_sparse.row)
    # print("label_sparse.col: ", label_sparse.col)
    indices = np.vstack((label_sparse.row, label_sparse.col))
    # print("indices: ", indices)

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = label_sparse.shape
    label_sparse = torch.sparse.LongTensor(i, v, torch.Size(shape))

    return label_sparse


def transform_label_matrix2spans(label_matrix, id2label_map=None):
    # Rm为l*l*c的tensor,表征着每一个起始i终止i的片段属于各个实体的概率
    # 将该矩阵解码为实体列表，每一个实体用tuple表示，（category_id, pos_b, pos_e）
    # 如果label标签为True，则输入为l*l的tensor
    entities = []
    cate_tensor = label_matrix
    cate_indices = torch.nonzero(cate_tensor)
    for index in cate_indices:
        cate_id = int(cate_tensor[index[0], index[1]])
        label_name = id2label_map[cate_id]
        entities.append((int(index[0]), int(index[1]), label_name))

    return entities


def Rm2entities(Rm, is_flat_ner=True, id2label_map=None):
    Rm = Rm.squeeze(0)

    # get score and pred l*l
    score, cate_pred = Rm.max(dim=-1)

    # fliter mask
    # mask category of none-entity
    seq_len = cate_pred.shape[1]
    zero_mask = (cate_pred == torch.tensor([0]).float().to(score.device))
    score = torch.where(zero_mask.byte(), torch.zeros_like(score), score)
    # pos_b <= pos_e check
    score = torch.triu(score)  # 上三角函数
    cate_pred = torch.triu(cate_pred)

    # get all entity list
    all_entity = []
    score_shape = score.shape
    score = score.reshape(-1)
    cate_pred = cate_pred.reshape(-1)
    entity_indices = (score != 0).nonzero(as_tuple = False).squeeze(-1)
    for i in entity_indices:
        i = int(i)
        score_i = float(score[i].item())
        cate_pred_i = int(cate_pred[i].item())
        pos_s = i // seq_len
        pos_e = i % seq_len
        all_entity.append((pos_s,pos_e,cate_pred_i,score_i))

    # sort by score
    all_entity = sorted(all_entity, key=lambda x:x[-1])
    res_entity = []

    for ns, ne, t, _ in all_entity:
        for ts, te, _ in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break

            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are allowed
                break

        else:
            res_entity.append((ns, ne, id2label_map[t]))
    return set(res_entity)


def count_same_entities(label_items, pred_items):
    count = 0
    for item in label_items:
        if item in pred_items:
            count += 1
    return count


def trans_span2bio(
    input_seq,
    max_seq_len=None,
    real_seq_len=None,
    list_entities=None,
):
    # 将 span 转化为 BIO 序列
    list_labels = ["O"] * max_seq_len
    # print(list_labels)
    # print("real_seq_len: ", real_seq_len)

    for s, e, lab in list_entities:
        list_labels[s] = "B-" + lab
        list_labels[s + 1 : e + 1] = ["I-" + lab] * (e - s)

    list_labels = list_labels[: int(real_seq_len)]
    return list_labels


def get_biaffine_table(Rm, list_spans):
    Rm = Rm.squeeze(0)

    score, cate_pred = Rm.max(dim=-1)

    # fliter mask
    # mask category of none-entity
    seq_len = cate_pred.shape[1]

    # tmp_label = np.zeros((seq_len, seq_len))
    biaffine_label = np.full((seq_len, seq_len), 'O', dtype=np.str)
    for span in list_spans:
        biaffine_label[span[0], span[1]] = span[2]

    return biaffine_label


# if __name__ == '__main__':
#     label_list = ["B-PER", "B-ORG",  "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-PER"]
#     a = label_list_to_entities(label_list)
#     print(a)


