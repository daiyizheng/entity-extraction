#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 21:48
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : models.py
from __future__ import annotations

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchcrf import CRF



class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class CRFLayer(nn.Module):
    def __init__(self, args):
        super(CRFLayer, self).__init__()
        self.num_labels = args.num_labels
        self.classifier = Classifier(
            2*args.hidden_dim,
            self.num_labels,
            args.dropout
        )  # 分类层，对每一个NER label进行预测

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(self, sequence_output, attention_mask, labels):
        # input_ids: (B, L)  batch_size * max_sequence_lenth
        # attention_mask: (B, L)
        # token_type_ids: (B, L)
        # slot_labels_ids: (B, L)

        slot_logits = self.classifier(sequence_output)  # (B, L, num_slot_labels) # num_slot_labels在每个token上对每个标签类别都进行一个打分
        outputs = (slot_logits,)

        if labels is not None:
            slot_loss = self.crf(
                slot_logits,
                labels,
                mask=attention_mask.byte(),
                reduction='mean'
            )
            slot_loss = -1 * slot_loss  # negative log-likelihood

            outputs = (slot_loss,) + outputs

        return outputs

class BiLstm(nn.Module):
    def __init__(self, args):
        super(BiLstm, self).__init__()
        self.args = args
        ## 是否使用预训练embedding
        if not args.random_init_w2v:
            w2v_matrix = np.asarray(args.vector_list)
            self.embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(torch.FloatTensor(w2v_matrix), freeze=False)
        else:# 不使用预训练
            self.embedding = nn.Embedding(len(args.vocab_list), args.embed_dim, padding_idx=args.word2id.get(args.pad_token, len(args.vocab_list)-1))
            # 初始化权重
            self._init_weight()
        self.encoder = nn.LSTM(args.embed_dim, args.hidden_dim, dropout=args.dropout, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(args.hidden_dim*2, args.hidden_dim, dropout=args.dropout, batch_first=True, bidirectional=True)
        self.classifier = Classifier(2 * args.hidden_dim, args.num_labels)
        if self.args.use_crf:
            self.crf_classifiers = CRFLayer(args)

    def _init_weight(self):
        ## embedding初始化
        torch.nn.init.uniform_(self.embedding.weight, -0.10, 0.10)
        # self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        # self.embedding.weight.data[self.embedding.padding_idx].zero_()
        ## 线性层初始化
        # self.linear.weight.data.normal_(mean=0.0, std=0.02)
        # self.linear.bias.data.zero_()
        ## LSTM初始化


    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None):
        emb_out = self.embedding(input_ids)
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_out, _ = self.encoder(emb_out)
        decoder_out, _ = self.decoder(encoder_out)#

        if self.args.use_crf:
            outputs = self.crf_classifiers(decoder_out, attention_mask, labels)
        else:
            logits = self.classifier(decoder_out)#[bz, max_len, num_class]

            outputs = (logits, )
            if labels is not None:
                active_loss = attention_mask.view(-1) == 1
                activate_logits = logits.view(-1, self.args.num_labels)[active_loss]
                activate_labels = labels.view(-1)[active_loss]
                loss = F.cross_entropy(activate_logits, activate_labels,
                                       ignore_index=self.args.pad_token_label_id or self.args.label_map.get("PAD", -100))
                outputs = (loss,) + outputs
        return outputs


