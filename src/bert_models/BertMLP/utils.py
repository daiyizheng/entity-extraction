#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 13:36
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : utils.py
from typing import Optional, Text, Any, Dict
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

# def sub_token_label_scheme(token_list:List,
#                            label_list:List, tokenizer:PreTrainedTokenizerBase,
#                            scheme_type:Text="v1"):
#     if scheme_type=="v1":
#         text, entities = conll_subword_label_alignment(token_list, label_list)
#         encoded, labels = subword_label_alignment(text, entities, tokenizer)
#     else:
#         raise NotImplementedError
#     return text, encoded["input_ids"], encoded["offset_mapping"], labels


