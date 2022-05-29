#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 21:25
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py
from __future__ import annotations

from src.ml_models.Hmm.hmm import HMM
from src.ml_models.Hmm.custome_tokenizers import WordTokenizerFast

MODEL_CLASSES = {
    "hmm": (None, HMM, WordTokenizerFast),
}