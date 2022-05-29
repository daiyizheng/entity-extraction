#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 23:14
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : finetuning_argparse.py.py

import argparse

def get_argparse():
    ## Required parameters
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=False,
                        help="name of the task.")
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--config_dir", default=None, type=str, required=True,
                        help="The parameter config path.")
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output dir where the model predictions and checkpoints will be written.")

    ## other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization.Sequence longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--do_lower_case", default=False,
                        help="Set this flag if you are using an uncased model.")

    ## dropout
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory.")
    parser.add_argument("--overwrite_cache", default=False,
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument("--log_dir", default='log/', type=str, required=False,
                        help="The log directions.")


    return parser