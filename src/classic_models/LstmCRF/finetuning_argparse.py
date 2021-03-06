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
    parser.add_argument("--is_flat_ner", default=False,
                        help="whether the task is flat, or not (nested).")
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--config_dir", default=None, type=str, required=True,
                        help="The parameter config path.")
    parser.add_argument("--vocab_dir", default=None, type=str, required=False,
                        help="vocab path.")
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output dir where the model predictions and checkpoints will be written.")

    ## other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same model_name.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do yu want to store the pre-trained models downloaded from s3.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization.Sequence longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--evaluate_during_training", default=True,
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=False,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--embed_dim", default=200,
                        help="dictionary mapping word vector size for Embedding layer")
    ## ????????????
    parser.add_argument("--vocab_size", default=0, type=int,
                        help="build vocab size for corpus")
    ## ???????????????
    parser.add_argument("--hidden_size", default=100, type=int,
                        help="hidden size for LSTM output")
    parser.add_argument("--use_crf", default=True, type=bool,
                        help="Whether to use crf layers")

    ## model configurations
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward.update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_layer_decay", default=0.9, type=float,
                        help="ratio of layerwise lr decay, i.e., lower layer gets smaller lr.")

    # output_dropout
    ## dropout
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for neural network")
    parser.add_argument("--tensorboardx_path", default=None, type=str,
                        help="tensorbox path")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epoches to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", default=50, type=int,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", default=50, type=int,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--metric_key_for_early_stop", default="f1", type=str,
                        help="metric name for early stopping ")
    parser.add_argument("--patience_for_early_stop", default=20, type=int,
                        help="patience number for early stopping ")
    parser.add_argument("--eval_all_checkpoints", default=False,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CuDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory.")
    parser.add_argument("--overwrite_cache", default=False,
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument("--gradient_checkpointing", default=True,
                        help="Whether to use gradient_checkpointing.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit mixed precision through NVIDIA apex instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", default="01", type=str,
                        help="For fp: Apex AMP optimization level selected in ['00', '01', '02', and '03'].")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", default="", type=str, help="For distant debugging.")
    parser.add_argument("--server_port", default="", type=str, help="For distant debugging.")

    #  embedding layer
    parser.add_argument("--embeddings_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for embedding layer.")

    # crf layer
    parser.add_argument("--crf_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for crf layer.")

    parser.add_argument("--classifier_learning_rate", default=1e-3, type=float,
                        help="The initial learning classifier for crf layer.")

    # Lstm
    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to use Lstm to extract the features")
    parser.add_argument("--lstm_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for LSTM layer.")

    parser.add_argument("--use_focal", action="store_true",
                        help="Whether to use focal loss")

    # pabee
    parser.add_argument("--patience", default=0, type=float, required=False, )
    parser.add_argument("--regression_threshold", default=0, type=float, required=False, )
    parser.add_argument("--log_dir", default='log/', type=str, required=False,
                        help="The log directions.")

    parser.add_argument("--ignore_index", default=-100, type=int, required=False, )

    parser.add_argument("--sub_token_label_scheme", default="v1",
                        type=str, required=False,
                        help="???subtoken????????????ner????????? "
                             "[v1] ??????I-yyy ?????????"
                             "[v2] ?????? O ?????????"
                             "[v3] ?????? X ?????????(loss ??????????????????)",
                        )

    return parser