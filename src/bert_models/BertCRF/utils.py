#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 10:53
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


