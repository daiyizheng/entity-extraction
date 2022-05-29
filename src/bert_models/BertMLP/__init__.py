#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py.py

from src.transformers import BertConfig, BertTokenizerFast

from src.bert_models.BertMLP.modeling_bert import BertForTokenClassificationWithPabeeMLP

MODEL_CLASSES = {
    "bertmlp": (BertConfig, BertForTokenClassificationWithPabeeMLP, BertTokenizerFast),
}