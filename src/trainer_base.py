#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 23:36
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer_base.py
import abc
import json
import os
import logging

import torch

logger = logging.getLogger(__file__)

class TrainerBase(abc.ABC):
    def __init__(self, args, tokenizer, config, model, train_dataset=None, dev_dataset=None, test_dataset=None, **kwargs):
        self.args = args
        self.tokenizer = tokenizer
        self.config = config
        self.model = model

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError

    def save_model(self):
        # Save models checkpoint (Overwrite)
        outputs_dir = os.path.join(self.args.output_dir, "checkpoint")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        if self.args.model_type == "nezha":
            json.dump(
                model_to_save.config.__dict__,
                open(os.path.join(outputs_dir, "config.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
            state_dict = model_to_save.state_dict()
            output_model_file = os.path.join(outputs_dir, "pytorch_model.bin")
            torch.save(state_dict, output_model_file)

        else:
            model_to_save.save_pretrained(outputs_dir)

        # Save training arguments together with the trained models
        self.tokenizer.save_pretrained(outputs_dir)
        torch.save(self.args, os.path.join(outputs_dir, 'training_args.bin'))
        logger.info("Saving models checkpoint to %s", outputs_dir)

    def load_model(self, model_class):
        # Check whether models exists
        outputs_dir = os.path.join(self.args.output_dir, "checkpoint")
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")
        try:
            if self.args.model_type == "nezha":
                output_model_file = os.path.join(outputs_dir, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(output_model_file, map_location=self.args.device))
            else:
                self.model = model_class.from_pretrained(outputs_dir,
                                                         config=self.config,
                                                         cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                                                         labels_list=self.args.labels_list,
                                                         args=self.args,
                                                         )
            self.model.to(self.args.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")

    @classmethod
    def load(cls, *args, **kwargs):
        return cls(*args, **kwargs)

