#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 23:40
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

def ent2token_spans(text, entity_list, tokenizer):
    """

    :param text: 原始文本
    :param entity_list: [{'id': 1, 'start': 0, 'end': 2, 'mention': 'EU', 'type': 'ORG'}]
    :param tokenizer: 分词器
    :return:
    """
    ent2token_spans = []
    inputs = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    '''
    {'input_ids': [6224, 7481, 2218, 6206, 6432, 10114, 8701, 9719, 8457, 8758], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'offset_mapping': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (9, 14), (15, 18), (18, 21), (21, 24)]}
    '''
    token2char_span_mapping = inputs["offset_mapping"]#每个切分后的token在原始的text中的起始位置和结束位置
    text2token = tokenizer.tokenize(text, add_special_tokens=False)
    #['见', '面', '就', '要', '说', 'say', 'hello', 'yes', '##ter', '##day']
    for ent_span in entity_list:
        ent = text[ent_span["start"]:ent_span["end"]]
        ent2token = tokenizer.tokenize(ent, add_special_tokens=False)

        # 然后将按字符个数标注的位置  修订 成 分完词 以token为个体的位置
        token_start_indexs = [i for i, v in enumerate(text2token) if v==ent2token[0]]
        token_end_indexs = [i for i, v in enumerate(text2token) if v==ent2token[-1]]

        # 分词后的位置 转为字符寻址 要和之前标的地址要一致 否则 就出错了
        token_start_index = list(filter(lambda x:token2char_span_mapping[x][0]==ent_span["start"], token_start_indexs))
        token_end_index = list(filter(lambda x:token2char_span_mapping[x][-1] == ent_span["end"], token_end_indexs))

        if len(token_start_index)==0 or len(token_end_index)==0:
            continue
            # 无法对应的token_span中
        token_span = {"id":ent_span["id"],
                      "start":token_start_index[0],
                      "end":token_end_index[0],
                      "mention":ent_span["mention"],
                      "type":ent_span["type"]}
        ent2token_spans.append(token_span)
    return text, inputs["input_ids"], token2char_span_mapping, ent2token_spans



