#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 10:54
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py

import logging
import os, sys
import json
from typing import List

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import torch
from src.transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np


from src.bert_models.BertCRF import MODEL_CLASSES
from src.bert_models.BertCRF.metrics_utils import compute_metrics, get_slot_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__file__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        ## 加载模型
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model_type.lower()]
        self.tokenizer = self.tokenizer_class.from_pretrained(args.model_name_or_path, return_offsets_mapping=True,
                                                              do_lower_case=args.do_lower_case,
                                                              cache_dir=args.cache_dir if args.cache_dir else None)
        self.config = self.config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=args.num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          output_dropout=args.output_dropout,
                                          gradient_checkpointing=args.gradient_checkpointing)

        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                            config=self.config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,
                                            labels_list=args.labels_list,
                                            args=args,
                                            )
        self.model.to(args.device)

        # tensorboardx
        self.tb_writer = SummaryWriter(args.tensorboardx_path)

        # for early  stopping
        self.global_epoch = 0
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.patience_for_early_stop = args.patience_for_early_stop
        self.early_stopping_counter = 0
        self.do_early_stop = False

    def train(self):
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)

        train_sampler = RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset, seed=self.args.seed)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size
        )
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        ## mdoel information
        # Prepare optimizer and schedule (linear warmup and decay)
        print("**********************************Prepare optimizer and schedule start************************")
        for n, p in self.model.named_parameters():
            print(n)
        print("**********************************Prepare optimizer and schedule middle************************")
        optimizer_grouped_parameters = []
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate
            },
            {
                'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.learning_rate
            }
        ]

        crf_param_optimizer = list(self.model.crf_classifiers.named_parameters())
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.crf_learning_rate
            },
            {
                'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.crf_learning_rate
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )
        print("**********************************Prepare optimizer and schedule end***************************")

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                              output_device=self.args.local_rank,
                                                              find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.args.train_batch_size * self.args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        temp_score = 0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            self.global_epoch = self.global_epoch + 1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)  # GPU or CPU

                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet"] else None

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logs = {}
                        if self.args.local_rank == -1 and self.args.evaluate_during_training:
                            _, results = self.evaluate()
                            temp_score = results[self.metric_key_for_early_stop]
                            logger.info("*" * 50)
                            logger.info("current step score for metric_key_for_early_stop: {}".format(temp_score))
                            logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                            logger.info("*" * 50)

                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value
                                print("eval_{}".format(key), value, global_step)

                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        # learning_rate_biaffine = scheduler.get_lr()[2]
                        logs["learning_rate"] = learning_rate_scalar
                        # logs["learning_rate_crf"] = learning_rate_biaffine
                        logs["loss"] = loss_scalar
                        print("lr", learning_rate_scalar, global_step)
                        print("loss", loss_scalar, global_step)
                        logging_loss = tr_loss
                        for key, value in logs.items():
                            self.tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                        ## 保存最有结果
                        if temp_score > self.best_score:
                            self.best_score = temp_score
                            self.early_stopping_counter = 0
                            if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                self.save_model()

                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.patience_for_early_stop: #patience_for_early_stop
                                self.do_early_stop = True

                                logger.info("best score is {}".format(self.best_score))

                        if self.do_early_stop:
                            break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.do_early_stop:
                    epoch_iterator.close()
                    break
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.do_early_stop:
                train_iterator.close()
                break
        if self.args.local_rank in [-1, 0]:
            self.tb_writer.close()
        return global_step, tr_loss / global_step

    def evaluate(self):

        if self.args.model_type == "albert":
            self.model.albert.set_regression_threshold(self.args.regression_threshold)
            self.model.albert.set_patience(self.args.patience)
            self.model.albert.reset_stats()
        elif self.args.model_type.lower() == "bertmlp":
            self.model.bert.set_regression_threshold(self.args.regression_threshold)
            self.model.bert.set_patience(self.args.patience)
            self.model.bert.reset_stats()
        elif self.args.model_type == "bertcrf":
            self.model.bert.set_regression_threshold(self.args.regression_threshold)
            self.model.bert.set_patience(self.args.patience)
            self.model.bert.reset_stats()
        else:
            raise NotImplementedError()

        dataset = self.dev_dataset
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset, shuffle=False, seed=self.args.seed)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        slot_preds = None
        out_slot_labels_ids = None
        out_token_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]
                          }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet"] else None

                outputs = self.model(**inputs) # (loss, logits )
                tmp_eval_loss, logits = outputs
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            # label prediction
            # Slot prediction
            if slot_preds is None:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = logits.detach().cpu().numpy()
                # slot_preds = np.array(model.crf.decode(logits))  # 维特比解码
                out_slot_labels_ids = inputs["labels"].detach().cpu().numpy()
                out_token_ids = inputs["input_ids"].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                out_token_ids = np.append(out_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)

        # logits 取 argmax， 得到预测的标签序号
        slot_preds = np.argmax(slot_preds, axis=2)  # (n, L, NUM_OF_LABELS) --> (n, L, 1)
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_label_map = {i: label for i, label in enumerate(self.args.labels_list)}
        print("slot_label_map: ", slot_label_map)

        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        for i in range(out_slot_labels_ids.shape[0]):  # 这个时候需要区分一个问题：经过bert之后的标签有很大一部分是我们不需要的东西
            for j in range(out_slot_labels_ids.shape[1]):
                token_id_ = out_token_ids[i][j]
                token_ = self.tokenizer.convert_ids_to_tokens([token_id_])[0]

                if token_ not in ["[CLS]", "[SEP]", "[PAD]"] and not token_.startswith("##"):
                    label_id_ = out_slot_labels_ids[i][j]
                    label_ = slot_label_map[label_id_]

                    # print("token: ", token_, out_token_ids[i][j], out_token_ids[i])
                    # print("label: ", slot_label_map[out_slot_labels_ids[i][j]])
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])  # 将这一部分去掉的话就是原句子的NER预测
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

                # if out_slot_labels_ids[i, j] != pad_token_label_id:  # 如果标签是 pad_token_label_id表明这个token不是我们想要的
                #     out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])  # 将这一部分去掉的话就是原句子的NER预测
                #     slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = get_slot_metrics(slot_preds_list, out_slot_label_list)  # 做评估和打分
        results.update(total_result)
        ### add
        ######
        logger.info("***** Eval results %s *****")
        for key in sorted(results.keys()):
            logger.info(" %s = %s", key, str(results[key]))


        # label prediction result
        if self.args.eval_all_checkpoints and self.args.patience != 0:
            if self.args.model_type == "albert":
                self.args.model.albert.log_stats()
            elif self.args.model_type == "bertmlp":  ### args.model_type == "bert":
                self.args.model.bert.log_stats()
            elif self.args.model_type == "bertcrf":  ### args.model_type == "bert":
                self.model.bert.log_stats()
            else:
                raise NotImplementedError()

        return eval_loss, results

    def predict(self)->List:
        if self.args.model_type == "albert":
            self.model.albert.set_regression_threshold(self.args.regression_threshold)
            self.model.albert.set_patience(self.args.patience)
            self.model.albert.reset_stats()
        elif self.args.model_type.lower() == "bertmlp":
            self.model.bert.set_regression_threshold(self.args.regression_threshold)
            self.model.bert.set_patience(self.args.patience)
            self.model.bert.reset_stats()
        else:
            raise NotImplementedError()
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        dataset = self.test_dataset
        test_sample = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset, shuffle=False, seed=self.args.seed)
        test_dataloader = DataLoader(dataset, sampler=test_sample,batch_size=self.args.eval_batch_size)
        logger.info("***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()

        slot_preds = None
        out_token_ids = None

        for batch in tqdm(test_dataloader, desc="Prediction"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet"] else None

                outputs = self.model(**inputs) # (loss, logits )
                logits = outputs[0]

            # label prediction
            # Slot prediction
            if slot_preds is None:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = logits.detach().cpu().numpy()
                # slot_preds = np.array(model.crf.decode(logits))  # 维特比解码
                out_token_ids = inputs["input_ids"].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, logits.detach().cpu().numpy(), axis=0)
                out_token_ids = np.append(out_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
        # logits 取 argmax， 得到预测的标签序号
        slot_preds = np.argmax(slot_preds, axis=2)  # (n, L, NUM_OF_LABELS) --> (n, L, 1)

        slot_label_map = {i: label for i, label in enumerate(self.args.labels_list)}
        print("slot_label_map: ", slot_label_map)

        slot_preds_list = [[] for _ in range(out_token_ids.shape[0])]

        for i in range(out_token_ids.shape[0]):  # 这个时候需要区分一个问题：经过bert之后的标签有很大一部分是我们不需要的东西
            for j in range(out_token_ids.shape[1]):
                token_id_ = out_token_ids[i][j]
                token_ = self.tokenizer.convert_ids_to_tokens([token_id_])[0]

                if token_ not in ["[CLS]", "[SEP]", "[PAD]"]:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        return slot_preds_list




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

    def load_model(self):
        # Check whether models exists
        outputs_dir = os.path.join(self.args.output_dir, "checkpoint")
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.config = self.config_class.from_pretrained(
                outputs_dir,
                num_labels=self.args.num_labels,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                output_dropout=self.args.output_dropout,
                gradient_checkpointing=self.args.gradient_checkpointing)

            self.tokenizer = self.tokenizer_class.from_pretrained(
                outputs_dir,
                return_offsets_mapping=True,
                do_lower_case=self.args.do_lower_case,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None)

            if self.args.model_type == "nezha":
                output_model_file = os.path.join(outputs_dir, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(output_model_file, map_location=self.args.device))
            else:
                self.model = self.model_class.from_pretrained(outputs_dir,
                                                              config=self.config,
                                                              cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                                                              labels_list=self.args.labels_list,
                                                              args=self.args,
                                                              )
            self.model.to(self.args.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")