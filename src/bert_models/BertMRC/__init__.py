#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 21:10
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py

from src.transformers import BertConfig, BertTokenizerFast

from src.bert_models.BertMRC.modeling_bert import BertForTokenClassificationWithPabeeMRC

MODEL_CLASSES = {
    "bertmrc": (BertConfig, BertForTokenClassificationWithPabeeMRC, BertTokenizerFast),
}