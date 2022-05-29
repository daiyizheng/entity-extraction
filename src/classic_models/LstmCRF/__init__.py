#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 21:25
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py

from __future__ import annotations

from src.classic_models.LstmCRF.modeling import BiLstm
from src.classic_models.LstmCRF.custome_tokenizers import WordTokenizerFast

MODEL_CLASSES = {

    "bilstm": (None, BiLstm, WordTokenizerFast),

}