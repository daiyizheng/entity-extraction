#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 21:24
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py.py
from src.bert_models.Biaffine.modeling_bert import BertForTokenClassificationBiaffine
from src.transformers import BertConfig, BertTokenizerFast

MODEL_CLASSES = {
    "bertbiaffine": (BertConfig, BertForTokenClassificationBiaffine, BertTokenizerFast),
}  # important

