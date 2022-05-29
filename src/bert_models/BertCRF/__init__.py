#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 21:24
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py

from src.transformers import BertTokenizerFast, BertConfig

from src.bert_models.BertCRF.modeling_bert import BertForTokenClassificationWithPabeeCRF

MODEL_CLASSES = {

    "bertcrf": (BertConfig, BertForTokenClassificationWithPabeeCRF, BertTokenizerFast),

}