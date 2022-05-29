#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 22:04
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py

import logging
import os, sys
import json
from typing import List

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import torch
from torch.nn import functional as F

from src.bert_models.BertMRC import MODEL_CLASSES
from src.bert_models.BertMRC.metrics.mrc_ner_evaluate import flat_ner_performance, nested_ner_performance, \
    update_label_lst, flat_transform_bmes_label
from src.transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np

# from src.bert_models.BertMRC.metrics_utils import compute_metrics, get_slot_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__file__)

class Trainer(object):
    def __init__(self, args,
                 config,
                 model,
                 tokenizer,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        ## 加载模型
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

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
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             'lr': self.args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': self.args.learning_rate}
        ]

        classifiers_param_optimizer = list(self.model.classifiers.named_parameters())
        optimizer_grouped_parameters += [
            {'params': [p for n, p in classifiers_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             'lr': self.args.classifiers_learning_rate
             },
            {'params': [p for n, p in classifiers_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': self.args.classifiers_learning_rate}
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
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
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
                          "start_position_ids": batch[3],
                          "end_position_ids": batch[4],
                          }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet", "bertmrc"] else None

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
                # print("loss: ", loss.item())
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
                    if self.args.local_rank in [-1,0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
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
                        # logs["learning_rate_crf"] = learning_rate_biaffine //batch_size跟学习率成反比
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
                            if self.early_stopping_counter >= self.patience_for_early_stop:  # patience_for_early_stop
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
        elif self.args.model_type.lower() == "bertmrc":
            self.model.bert.set_regression_threshold(self.args.regression_threshold)
            self.model.bert.set_patience(self.args.patience)
            self.model.bert.reset_stats()
        else:
            raise NotImplementedError()

        dataset = self.dev_dataset
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset,
                                                                                                        shuffle=False,
                                                                                                        seed=self.args.seed)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        start_pred_lst = []
        end_pred_lst = []

        start_gold_lst = []
        end_gold_lst = []
        ner_cate_lst = []

        span_pred_lst = []
        span_gold_lst = []
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "start_position_ids": batch[3],
                          "end_position_ids": batch[4],
                          }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet", "bertmrc"] else None

                outputs = self.model(**inputs)  # (loss, start_logits, endlogits )
                tmp_eval_loss, start_logits, end_logits = outputs
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            attention_mask = inputs["attention_mask"]
            start_logits = torch.argmax(start_logits, dim=-1) #[bz, seq_len]
            end_logits = torch.argmax(end_logits, dim=-1) #[bz, seq_len]

            ## mask 去除padding
            mask_len = attention_mask.sum(-1)
            for idx, length in enumerate(mask_len):
                start_position_ids = inputs['start_position_ids'][idx][:length].to("cpu").numpy().tolist()
                end_position_ids = inputs['end_position_ids'][idx][:length].to("cpu").numpy().tolist()
                pre_start_logits = start_logits.to("cpu")[idx][:length].numpy().tolist()
                pre_end_logits = end_logits.to("cpu")[idx][:length].numpy().tolist()
                start_pred_lst.append(pre_start_logits)
                end_pred_lst.append(pre_end_logits)
                start_gold_lst.append(start_position_ids)
                end_gold_lst.append(end_position_ids)
                span_pred_lst.append([[1] * len(pre_start_logits)] * len(pre_start_logits))
                span_gold_lst.append([[1] * len(start_position_ids)] * len(start_position_ids))

            ner_cate_ids = batch[5].to("cpu").numpy().tolist()
            ner_cate_lst+=ner_cate_ids



        if self.args.is_flat_ner:
            acc, precision, recall, f1, pred_list = flat_ner_performance(
                start_pred_lst,
                end_pred_lst,
                span_pred_lst,
                start_gold_lst,
                end_gold_lst,
                span_gold_lst,
                ner_cate_lst,
                self.args.labels_list,
                threshold=self.args.entity_threshold,
                dims=2)
        else:
            acc, precision, recall, f1, pred_list = nested_ner_performance(
                start_pred_lst,
                end_pred_lst,
                span_pred_lst,
                start_gold_lst,
                end_gold_lst,
                span_gold_lst,
                ner_cate_lst,
                self.args.labels_list,
                threshold=self.args.entity_threshold,
                dims=2)

        eval_loss = eval_loss / nb_eval_steps

        results = {
            "loss": eval_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        ### add
        logger.info("***** Eval results %s *****")
        for key in sorted(results.keys()):
            logger.info(" %s = %s", key, str(results[key]))

        if self.args.file_to_predict:  # 预测结果的记录，是不是可以解码成英文字符之后再输出？
            json.dump(
                pred_list,
                open(self.args.file_to_predict, 'w', encoding="utf-8"),
                ensure_ascii=False,
            )

        if self.args.eval_all_checkpoints and self.args.patience != 0:
            if self.args.model_type == "albert":
                self.model.albert.log_stats()
            elif self.args.model_type == "bertmrc":
                self.model.bert.log_stats()
            else:
                raise NotImplementedError()
        return eval_loss, results

    def predict(self) -> List:
        if self.args.model_type == "albert":
            self.model.albert.set_regression_threshold(self.args.regression_threshold)
            self.model.albert.set_patience(self.args.patience)
            self.model.albert.reset_stats()
        elif self.args.model_type.lower() == "bertmrc":
            self.model.bert.set_regression_threshold(self.args.regression_threshold)
            self.model.bert.set_patience(self.args.patience)
            self.model.bert.reset_stats()
        else:
            raise NotImplementedError()
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        dataset = self.test_dataset
        test_sample = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset,
                                                                                                       shuffle=False,
                                                                                                       seed=self.args.seed)
        test_dataloader = DataLoader(dataset, sampler=test_sample, batch_size=self.args.eval_batch_size)
        logger.info("***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()
        slot_preds_list = []
        pre_start_logits = []
        pre_end_logits = []
        ner_cate_lst = []
        span_pred_lst = []

        for batch in tqdm(test_dataloader, desc="Prediction"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1]
                          }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet", "bertmrc"] else None
                outputs = self.model(**inputs)  # (start_logits , end_logits)
                start_logits, end_logits = outputs

                attention_mask = inputs["attention_mask"]
                start_logits = torch.argmax(start_logits, dim=-1)  # [bz, seq_len]
                end_logits = torch.argmax(end_logits, dim=-1)
                ## mask 去除padding
                mask_len = attention_mask.sum(-1)
                for idx, length in enumerate(mask_len):
                    pre_start_logits.append(start_logits[idx][:length].to("cpu").numpy().tolist())
                    pre_end_logits.append(end_logits[idx][:length].to("cpu").numpy().tolist())
                    span_pred_lst.append([[1] * len(pre_start_logits)] * len(pre_start_logits))
                ner_cate_ids = batch[5].to("cpu").numpy().tolist()
                ner_cate_lst+=ner_cate_ids


            cate_idx2label = {idx: value for idx, value in enumerate(self.args.labels_list)}
            # label_list: ["PER", "ORG", "O", "LOC"]
            up_label_lst = update_label_lst(self.args.labels_list)
            # up_label_lst: ["B-PER", "I-PER", "E-PER", "S-PER", ]
            label2idx = {label: i for i, label in enumerate(up_label_lst)}

            for pred_start, pred_end, pred_span, ner_cate_item in zip(pre_start_logits, pre_end_logits, span_pred_lst, ner_cate_lst):
                ner_cate = cate_idx2label[ner_cate_item]
                pred_bmes_label = flat_transform_bmes_label(pred_start, pred_end, pred_span, ner_cate, threshold=self.args.entity_threshold)
                slot_preds_list.append(pred_bmes_label)
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
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type.lower()]
        outputs_dir = os.path.join(self.args.output_dir, "checkpoint")
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.config = config_class.from_pretrained(
                outputs_dir,
                num_labels=self.args.num_labels,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                output_dropout=self.args.output_dropout,
                gradient_checkpointing=self.args.gradient_checkpointing)

            self.tokenizer = tokenizer_class.from_pretrained(
                outputs_dir,
                return_offsets_mapping=True,
                do_lower_case=self.args.do_lower_case,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None)

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