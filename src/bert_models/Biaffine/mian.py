#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/17 13:14
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : mian.py

import yaml, json
import os,sys
sys.path.append("./")
import logging
from datetime import datetime

import torch

from src.bert_models.Biaffine.trainer import Trainer
from src.bert_models.Biaffine.data_loader import load_and_cache_examples, ResultExample
from src.bert_models.Biaffine.finetuning_argparse import get_argparse
from src.bert_models.Biaffine import MODEL_CLASSES
from src.bert_models.Biaffine.utils import override_defaults, seed_everything, get_labels, init_logger


logger = logging.getLogger(__file__)

def main():
    args = get_argparse().parse_args()
    ## 修正的参数
    f = open(args.config_yml_dir, "r")
    fix_args = yaml.full_load(f)
    for k in fix_args:
        args = override_defaults(args, fix_args[k])

    ## 随机种子
    seed_everything(args.seed)

    ## 模型输出目录
    args.output_dir = os.path.join(args.output_dir, args.model_type.lower()+"_"+ args.task_name + "_" + str(args.seed))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## 日志输出
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ## tensorboardx
    args.tensorboardx_path = os.path.join(args.tensorboardx_path,
                                          args.model_type.lower() + "_" + args.task_name + "_" + str(
                                              args.seed) + "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(args.tensorboardx_path):
        os.makedirs(args.tensorboardx_path)

    # Setup CUDA， GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})
    else:  # initializes the distributed backend which will take care of suchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_porcess_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        filename=os.path.join(args.log_dir, args.model_type.lower() + "_" + args.task_name + "_" + str(args.seed)+".log"))

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    ## 加载模型
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type.lower()]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, return_offsets_mapping=True,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    labels_list = get_labels(args.labels)
    print("labels: ", labels_list)
    label_map = {label: i for i, label in enumerate(labels_list)}
    num_labels = len(labels_list)
    args.num_labels = num_labels
    args.label_map = label_map
    args.labels_list = labels_list


    logger.info("Training/evaluation parameters %s", args)
    ##--------------------------------- 加载数据 -----------------------------------------
    train_dataset, _ = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, _ = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = []

    print("train_dataset: ", len(train_dataset))
    print("dev_dataset: ", len(dev_dataset))
    print("test_dataset: ", len(test_dataset))

    ##------------------- 加载训练 ------------------------------
    trainer = Trainer(
        args, train_dataset, dev_dataset, test_dataset
    )

    if args.do_train:
        global_step, tr_loss = trainer.train()
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        trainer.load_model()
        loss, result = trainer.evaluate()
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

    if args.do_predict:
        trainer.load_model()


if __name__ == '__main__':
    main()