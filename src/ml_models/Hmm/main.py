#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 0:48
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : main.py
import os
import logging
import yaml

from src.ml_models.Hmm import MODEL_CLASSES
from src.ml_models.Hmm.data_loader import load_and_cache_examples
from src.ml_models.Hmm.finetuning_argparse import get_argparse
from src.ml_models.Hmm.metrics_utils import get_slot_metrics
from src.ml_models.Hmm.utils import override_defaults, seed_everything, get_labels


def main():
    args = get_argparse().parse_args()
    ## 修正的参数
    f = open(args.config_dir, "r")
    fix_args = yaml.full_load(f)
    for k in fix_args:
        args = override_defaults(args, fix_args[k])

    ## 随机种子
    seed_everything(args.seed)

    ## 模型输出目录
    args.output_dir = os.path.join(args.output_dir, args.model_type.lower() + "_" + args.task_name + "_" + str(args.seed))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## 日志输出
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    labels_list = get_labels(args.labels)
    print("labels: ", labels_list)
    label_map = {label: i for i, label in enumerate(labels_list)}
    num_labels = len(labels_list)
    args.num_labels = num_labels
    args.label_map = label_map
    args.labels_list = labels_list

    # setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        filename=os.path.join(args.log_dir, args.model_type.lower() + "_" + args.task_name + "_" + str(args.seed) + ".log"))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type.lower()]
    tokenizer = tokenizer_class.from_pretrained(do_lower_case=args.do_lower_case)
    config = config_class
    model = model_class(args=args)

    ##--------------------------------- 加载数据 -----------------------------------------
    train_dataset, _ = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, _ = load_and_cache_examples(args, tokenizer, mode="dev")

    if args.do_train:
        model.train(train_dataset[0],train_dataset[1])
        ## model 验证
        predict_labels = model.predict(dev_dataset[0])
        results = get_slot_metrics(preds=predict_labels, labels=dev_dataset[1])
        ### add
        print("***** Eval results %s *****")
        for key in sorted(results.keys()):
            print(" %s = %s"%(key, str(results[key])))

        model.save_model()

    if args.do_predict:
        model.load_model()
        test_dataset, _ = load_and_cache_examples(args, tokenizer, mode="test")
        predict_labels = model.predict(test_dataset[0])



if __name__ == '__main__':
    main()