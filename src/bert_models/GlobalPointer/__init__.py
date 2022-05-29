#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 10:37
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py.py
from src.bert_models.GlobalPointer.modeling_bert import BertForTokenClassificationGlobalPointer
from src.transformers import BertConfig, BertTokenizerFast

MODEL_CLASSES = {
    "bertglobalpointer": (BertConfig, BertForTokenClassificationGlobalPointer, BertTokenizerFast),
}