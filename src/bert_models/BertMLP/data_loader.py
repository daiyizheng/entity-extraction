#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 15:21
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : data_loader.py
from __future__ import annotations


import copy
import json
import os
import logging
from typing import List, Text, Optional

import torch

from liyi_cute.processor.cut_sentence import CutSentence
from liyi_cute.processor.data_convert import DataConvert
from liyi_cute.shared.imports.schemas.schema import Example, NerExample
from torch.utils.data import TensorDataset

from liyi_cute.shared.loadings import parser

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        sentence1: list. The words of the sequence.
        sentence2: list. The words of the sequence.
        label: (Optional) string. The label of the example.
    """
    def __init__(self, id, text, entities):
        self.id = id
        self.text = text
        self.entities = entities

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertMlpProcessor(object):
    """Processor for the BERT data set """

    def __init__(self, args):
        self.args = args

    @staticmethod
    def _read_file(data_path: Text, data_type:Text, task_name:Text):
        """Reads a tab separated value file."""
        examples = parser(dirname=data_path, fformat=data_type, task_name=task_name)

        return examples

    def get_examples(self,
                     path: Text,
                     mode: Text,
                     data_type: Text,
                     task_name: Text
                     ) -> Optional[List[Example], List[NerExample]]:

        logger.info("LOOKING AT {}".format(mode))
        if data_type == "conll":
            data_path = os.path.join(path, "{}.txt".format(mode))
        elif data_type == "json":
            data_path = os.path.join(path, "{}.json".format(mode))
        elif data_type == "ann":
            raise NotImplementedError
        else:
            raise NotImplementedError
        examples = self._read_file(data_path, data_type, task_name)
        return examples

processors = {
    'bertmlp': BertMlpProcessor,
}

def convert_examples_to_features(examples:Optional[List[Example], List[NerExample]],
                                 args,
                                 tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 special_tokens_count=2,
                                 pad_token_label_id=-100,
                                 ):
    # Setting based on the current models type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    max_seq_length = args.max_seq_length
    label_map = args.label_map
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize
        # token = tokenizer.tokenize(example.words)
        text = example.text
        entities = example.entities

        if isinstance(text, str):
            text, tokens, token_span, offset_mapping = DataConvert.span_subword_token(text, entities, tokenizer)
            input_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
            labels = ["O"] * len(tokens)
            for token_start_index, token_end_index, token_type in token_span:
                labels[token_start_index] = "B-" + token_type
                for tk_id in range(token_start_index+1, token_end_index+1):
                    labels[tk_id] = "I-"+ token_type

        elif isinstance(text, list):
            if args.token_subword:
                text, tokens, labels, offset_mapping = DataConvert.conll_subword_token(text, entities, tokenizer)
                input_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
            else:
                input_ids = [tokenizer.convert_tokens_to_ids(t) for t in text]
                offset_mapping = [1] * len(input_ids)
                labels = entities
        else:
            raise NotImplementedError

        tokens = [tokenizer.convert_ids_to_tokens(tk) for tk in input_ids]

        labels = [label_map[l] for l in labels]

        # Account for [CLS] and [SEP]
        if len(input_ids) > (max_seq_length - special_tokens_count):
            input_ids = input_ids[:(max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            offset_mapping = offset_mapping[:(max_seq_length - special_tokens_count)]


        # Add [SEP] token
        tokens = tokens + [sep_token]
        input_ids = input_ids + [tokenizer.sep_token_id]
        labels = labels + [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * (len(input_ids))

        # Add [CLS] token
        tokens = [cls_token] + tokens
        input_ids = [tokenizer.cls_token_id] + input_ids
        labels = [pad_token_label_id] + labels
        token_type_ids = [cls_token_segment_id] + token_type_ids
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        labels = labels + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_length)
        assert len(labels) == max_seq_length, "Error with slot labels length {} vs {}".format(len(labels), max_seq_length)


        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.id)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("slot_labels: %s" % " ".join([str(x) for x in labels]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                slot_labels_ids=labels
            )
        )
        setattr(example, "offset_mapping", offset_mapping)

    return features, examples

def load_and_cache_examples(args, tokenizer, mode):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[args.model_type](args)

    cached_examples_file = os.path.join(
        os.path.join(args.data_dir, args.data_name, args.data_type),
        "cache_examples_{}_{}_{}_{}_{}".format(
            args.model_type,
            args.data_name,
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)
        )
    )

    if os.path.exists(cached_examples_file) and not args.overwrite_cache:
        logger.info("Loading examples from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode, args.data_type, args.task_name)
        elif mode == "dev":
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode, args.data_type, args.task_name)
        elif mode == "test":
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode, args.data_type, args.task_name)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        if args.local_rank in [-1, 0]:
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
    cut = CutSentence()
    print("samples length before cut: ", len(examples) )
    examples = cut(examples=examples, is_fine_cut=args.is_fine_cut, is_chinese=args.is_chinese)
    print("samples length after cut: ", len(examples))
    features, examples = convert_examples_to_features(
        examples,
        args,
        tokenizer,
        special_tokens_count=2)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
    )

    return dataset, examples