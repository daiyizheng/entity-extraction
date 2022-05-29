#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 16:16
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : modeling_bert.py
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from src.bert_models.GlobalPointer.loss_function_utils import multilabel_categorical_crossentropy
from src.transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from src.transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertEncoder,
    BertModel,
    BertPreTrainedModel,
)

logger = logging.getLogger(__file__)

# *** add ***
class BertEncoderWithPabee(BertEncoder):
    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])

        hidden_states = layer_outputs[0]

        return hidden_states

class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_labels, use_RoPE):
        super(GlobalPointer, self).__init__()
        self.inner_dim = 64 ##中间维度层
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.num_labels*self.inner_dim*2)
        self.use_RoPE = use_RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        ## 相对位置编码
        # PE(pos, 2i) = sin(\frac{pos}{1000^{\frac{2i}{d_{model}}}})
        # PE(pos, 2i+1) = cos(\frac{pos}{1000^{\frac{2i+1}{d_{model}}}})
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)#[max_len, 1]
        indices = torch.arange(0, output_dim//2, dtype=torch.float, device=device)
        indices = torch.pow(1000, -2*indices/output_dim)
        embeddings = position_ids * indices#[max_len, output_dim//2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) # #[max_len, inner_dim//2, 2]
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape)))) # 扩充了一维batch_size [bz, max_seq, inner_dim//2, 2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def forward(self, sequence_output, attention_mask):
        device = sequence_output.device
        batch_size, seq_len, hiden_size = sequence_output.size() # (batch_size, max_len, hidden_size)

        outputs = self.dense(sequence_output)# [bz, max_len, 2*num_class*inner_dim]

        outputs = torch.split(outputs, self.inner_dim*2, dim=-1) #inner_dim*2*[bz, max_len, num_class]
        outputs = torch.stack(outputs, dim=2)# 交叉融合，扩充一个维度[bz, max_len, nums_class, inner_dim*2]

        # 类似于dense生成q向量和k向量
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        # 维度是qw, kw[bz, max_len, num_class, inner_dim]

        if self.use_RoPE:
            # 相当于给q和k加入相对位置信息
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)#[bz, max_len, inner_dim]
            ## 旋转式位置编码
            #(q_0, q_1, q_3.....q_{d-2}, q_{d-1})^T \dot (cos(m\theta_0), cos(m\theta_0), cos(m\theta_1), cos(m\theta_1)...cos(m\theta_{d/2-1},cos(m\theta_{d/2-1}))^T
            #(-q_1, q_0, -q_3, q_2, ....-q_{d-1}, q_{d-2})^T \dot (sin(m\theta_0), sin(m\theta_0), sin(m\theta_1), sin(m\theta_1)...sin(m\theta_{d/2-1},sin(m\theta_{d/2-1}))^T
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 取奇数部分 None在该位置添加一个维度 值为1， [bz, max_len, 1, inner_dim//2]-> # repeat_interleave 指定维度重复次数 [bz, max_len, 1, inner_dim]
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)  # 取偶数部分  [bz, max_len, 1, inner_dim]
            qw2 = torch.stack([-qw[...,1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)  # [bz, max_len, num_label, inner_dim]
            qw = qw*cos_pos + qw2*sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) # #[bz, num_class, max_len, max_len]
        # [bz, max_len]->[bz, 1, max_len]->[bz, 1, 1, max_len] -> [bz, num_class, max_len, max_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12  # 1e12的意思是0为位置补上1e12

        ## 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask*1e12
        # print(logits.size())   # torch.Size([2, 10, 512, 512])
        return logits / self.inner_dim ** 0.5  # 类似attention的根号d

@add_start_docstrings(
    "The bare Bert Model transformer with PABEE outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModelWithPabee(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoderWithPabee(config)

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        logger.info(message)
        print(message)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
        labels_list=None,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        res = [output_layers[0](encoder_outputs[0], attention_mask)]

        return res

class BertForTokenClassificationGlobalPointer(BertPreTrainedModel):
    def __init__(self, config, labels_list, args, ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.labels_list = labels_list
        self.args = args

        self.bert = BertModelWithPabee(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.global_pointer_classifiers = nn.ModuleList(
            [
                GlobalPointer(
                    config.hidden_size,
                    args.num_labels,
                    use_RoPE=args.use_RoPE if args.use_RoPE else False
                ) for _ in range(1)
            ]
        )
        self.use_focal = getattr(config, "use_focal", False)
        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples::

            from transformers import BertTokenizer, BertForSequenceClassification
            from pabee import BertForSequenceClassificationWithPabee
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassificationWithPabee.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)

            loss, logits = outputs[:2]

        """
        logits = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.global_pointer_classifiers,
            regression=self.num_labels == 1,
            labels_list=self.labels_list,
        )

        outputs = (logits[-1],)  # 最后一层的output: batch_size * seq * seq * label_num
        #logits [bz, num_class, max_len, max_len]
        if labels is not None:

            loss_fct = multilabel_categorical_crossentropy()
            batch_size, ent_type_size = logits[-1].shape[:2]
            y_true = labels.reshape(batch_size * ent_type_size, -1)
            y_pred = logits[-1].reshape(batch_size * ent_type_size, -1)
            loss = loss_fct(y_pred, y_true)
            outputs = (loss,) + (outputs[0],)

        return outputs


