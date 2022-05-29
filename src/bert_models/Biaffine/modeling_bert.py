# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


from abc import ABC
###
import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from src.bert_models.Biaffine.loss_function_utils import FocalLoss
from src.transformers.activations import gelu_new
from src.transformers.file_utils import add_start_docstrings, \
    add_start_docstrings_to_model_forward

from src.transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertEncoder,
    BertModel,
    BertPreTrainedModel,
)


logger = logging.getLogger(__name__)


def get_useful_ones(out, label, attention_mask, sampling_ratio=None):
    # get mask, mask the padding and down triangle

    attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.shape[-1], -1)
    attention_mask = torch.triu(attention_mask)

    # flatten
    mask = attention_mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)

    # 同时，为了防止loss太不均衡，每次随机采样一部分作为负例
    sampling_mask = sampling_ratio * torch.ones_like(tmp_label).to(tmp_label.device)
    sampling_mask = torch.bernoulli(sampling_mask)
    sampling_mask = tmp_label + sampling_mask

    # index select, for gpu speed
    indices = (mask * sampling_mask).nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label.long()


class FeedForwardLayer(nn.Module, ABC):
    '''A two-feed-forward-layer module'''

    def __init__(self, d_in, d_hid, dropout=0.3):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(self.dropout(gelu_new(self.w_1(x))))
        x = residual + self.dropout2(x)
        return x


class BiaffineLayer(nn.Module, ABC):
    def __init__(self, inSize1, inSize2, classSize, dropout=0.3):
        super(BiaffineLayer, self).__init__()
        self.bilinearMap = nn.Parameter(
            torch.FloatTensor(
                inSize1 + 1,
                classSize,
                inSize2 + 1
            )
        )
        self.classSize = classSize

    def forward(self, x1, x2):
        # [b, n, v1] -> [b*n, v1]
        # print("BIAFFINEPARA:", self.bilinearMap)
        batch_size = x1.shape[0]
        bucket_size = x1.shape[1]

        x1 = torch.cat((x1,torch.ones([batch_size, bucket_size, 1]).to(x1.device)), axis=2)
        x2 = torch.cat((x2, torch.ones([batch_size, bucket_size, 1]).to(x2.device)), axis=2)
        # Static shape info
        vector_set_1_size = x1.shape[-1]
        vector_set_2_size = x2.shape[-1]

        # [b, n, v1] -> [b*n, v1]
        vector_set_1 = x1.reshape((-1, vector_set_1_size))

        # [v1, r, v2] -> [v1, r*v2]
        bilinear_map = self.bilinearMap.reshape((vector_set_1_size, -1))

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = torch.matmul(vector_set_1, bilinear_map)

        # [b*n, r*v2] -> [b, n*r, v2]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size * self.classSize, vector_set_2_size))

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        bilinear_mapping = torch.matmul(bilinear_mapping, x2.transpose(1, -1))

        # [b, n*r, n] -> [b, n, r, n]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size, self.classSize, bucket_size))
        # bilinear_mapping = torch.einsum('bxi,ioj,byj->bxyo', x1, self.bilinearMap, x2)
        return bilinear_mapping.transpose(-2, -1)


class BiaffineLayerv1(nn.Module, ABC):
    def __init__(self, inSize1, inSize2, classSize, dropout=0.3, device=None):
        super(BiaffineLayerv1, self).__init__()
        assert inSize1 == inSize2

        self.mlp = nn.Linear(4 * inSize1, classSize)

        self.classSize = classSize

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        seq_len = x1.shape[1]
        hidden_size = x1.shape[2]

        x1 = x1.view(batch_size, seq_len, 1, hidden_size).repeat(1, 1, seq_len, 1)
        x2 = x2.view(batch_size, 1, seq_len, hidden_size).repeat(1, seq_len, 1, 1)

        feat1 = x1 * x2  # elementwise乘积
        feat2 = torch.abs(x1 - x2)

        feat = torch.cat([x1, x2, feat1, feat2], dim=-1)

        output = self.mlp(feat)
        return output


# *** add ***
class BertEncoderWithPabee(BertEncoder):
    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])

        hidden_states = layer_outputs[0]

        return hidden_states


@add_start_docstrings(
    """Bert Model transformer with PABEE and a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BiaffineClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

        use_lstm = getattr(config, "use_lstm", False)
        self.lstm = None
        if use_lstm:
            from torch.nn import LSTM
            self.lstm = LSTM(
                config.hidden_size,
                config.hidden_size // 2,
                batch_first=True,
                bidirectional=True
            )
        self.use_focal = getattr(config, "use_focal", False)

        # 是否需要这两个前馈层
        use_ffn = getattr(config, "use_ffn", False)
        self.feedStart = None
        self.feedEnd = None
        if use_ffn:
            self.feedStart = FeedForwardLayer(
                config.hidden_size,
                config.hidden_size,
                dropout=config.hidden_dropout_prob
            )
            self.feedEnd = FeedForwardLayer(
                config.hidden_size,
                config.hidden_size,
                dropout=config.hidden_dropout_prob
            )

        if config.simplify_biaffine:
            self.biaffine = BiaffineLayerv1(
                config.hidden_size,
                config.hidden_size,
                self.num_labels,
                dropout=config.hidden_dropout_prob
            )
        else:
            self.biaffine = BiaffineLayer(
                config.hidden_size,
                config.hidden_size,
                self.num_labels,
                dropout=config.hidden_dropout_prob
            )

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(self,sequence_output):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        sequence_output = self.dropout(sequence_output)
        residual = sequence_output

        if self.lstm is not None:
            lstm_output, (_, _) = self.lstm(sequence_output)
            sequence_output = residual + lstm_output

            sequence_output = self.output_dropout(sequence_output)

        # 前馈层
        if self.feedStart is not None and self.feedEnd is not None:
            start = self.feedStart(sequence_output)
            end = self.feedEnd(sequence_output)
        else:
            start = sequence_output
            end = sequence_output

        score = self.biaffine(start, end)
        return score  # (loss), score, (hidden_states), (attentions)


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
        res = [output_layers[0](encoder_outputs[0])]

        return res


class BertForTokenClassificationBiaffine(BertPreTrainedModel):
    def __init__(self,config, labels_list, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.sampling_ratio = config.sampling_ratio

        self.do_hard_negative_sampling = config.do_hard_negative_sampling
        self.hns_multiplier = config.hns_multiplier

        self.labels_list = labels_list

        self.bert = BertModelWithPabee(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.biaffine_classifiers = nn.ModuleList(
            [BiaffineClassification(config) for _ in range(1)]
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
            output_layers=self.biaffine_classifiers,
            regression=self.num_labels == 1,
            labels_list=self.labels_list,
        )

        outputs = (logits[-1],)  # 最后一层的output: batch_size * seq * seq * label_num

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):

                if self.do_hard_negative_sampling:

                    # 得到有效的训练部分
                    tmp_out, tmp_label = get_useful_ones(
                        logits_item,
                        labels.to_dense(),
                        attention_mask,
                        sampling_ratio=0.1
                    )

                    # 每次训练,应该将 hard negative 与 一般负样本混合
                    score_, category_pred_ = tmp_out.max(dim=-1)

                    # print("tmp_out: ", tmp_out.shape)
                    # print("category_pred_: ", category_pred_.shape)

                    # hard negative: 非实体部分，实体得分高
                    pos_example = tmp_label != 0

                    # print("tmp_label == 0: ", (tmp_label == 0).shape)
                    # print("category_pred_ != 0: ", (category_pred_ != 0).shape)

                    neg_example = ( (tmp_label == 0) & (category_pred_ != 0) )
                    # print("neg_example: ", neg_example.shape)
                    # print("neg_example: ", neg_example.float().sum())
                    if neg_example.float().sum() == 0:
                        tmp_out_final = tmp_out
                        tmp_label_final = tmp_label

                    else:

                        pos_num = pos_example.sum()
                        neg_num = neg_example.sum()

                        # print("pos_num: ", pos_num)
                        # print("neg_num: ", neg_num)

                        # 采用 score 对 负例进行排序: 非实体部分，而且非实体类别得分还低
                        neg_scores_ = torch.masked_select(
                            tmp_out[:, 0],
                            neg_example.view(-1, 1).bool()
                        )
                        neg_scores_sort_, _ = torch.sort(neg_scores_, descending=True)
                        score_threshold = neg_scores_[int(pos_num * self.hns_multiplier) + 1]
                        # print("score_threshold: ", score_threshold)

                        hard_neg_example = tmp_out[:, 0] > score_threshold
                        cond = pos_example | hard_neg_example
                        hns_mask_idx = torch.where(cond, 1, 0)

                        tmp_out_final = tmp_out.index_select(
                            0, hns_mask_idx
                        )
                        tmp_label_final = tmp_label.index_select(
                            0, hns_mask_idx
                        )

                else:

                    tmp_out_final, tmp_label_final = get_useful_ones(
                        logits_item,
                        labels.to_dense(),
                        attention_mask,
                        sampling_ratio=self.sampling_ratio if self.training else 1,
                    )

                if self.use_focal:
                    loss_fct = FocalLoss()
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(tmp_out_final, tmp_label_final)

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs

        return outputs