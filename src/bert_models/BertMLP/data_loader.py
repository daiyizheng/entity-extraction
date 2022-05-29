#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 15:21
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : data_loader.py
from __future__ import annotations

import copy
import dataclasses
import json
import os,sys
import logging
import math
from typing import List, Dict, Tuple, Union, Text

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.preprocess_base import PreProcessBase
from src.schema import Entity

logger = logging.getLogger(__name__)

class ResultExample(object):
    def __init__(self, id, text, offsets_mapping, entities=[], pre_labels=[]):
        self.id = id
        self.text = text
        self.offsets_mapping = offsets_mapping
        self.entities = entities
        self.pre_labels =pre_labels

    @staticmethod
    def postion_to_string(pre_tags, entitie, offset_mapping, text, ids):
        tag_type = pre_tags[entitie[0]].split("-")[1]
        start = offset_mapping[entitie[0]][0]
        end = offset_mapping[entitie[-1]][1]
        return {"id": "T" + str(ids), "mention": text[start:end], "start": start, "end": end, "type": tag_type}

    def entity_to_json(self):
        entities = []
        entitie = []
        ids = 1
        for index, tag in enumerate(self.pre_labels):
            if tag[0] == "B" and len(entitie) == 0:
                entitie.append(index)
            elif tag[0] == "B" and len(entitie) != 0:
                entities.append(self.postion_to_string(self.pre_labels, entitie,
                                                       self.offsets_mapping, self.text, ids))
                ids += 1
                entitie = [index]
            elif tag[0] == "I":
                entitie.append(index)
            else:
                if entitie:
                    entities.append(self.postion_to_string(self.pre_labels, entitie,
                                                           self.offsets_mapping, self.text, ids))
                    ids += 1
                entitie = []

        return {"text": self.text, "pre_entities": entities, "entities":self.entities}

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

    @classmethod
    def _read_file(cls, input_file, set_type=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            tags = []
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    line_list = line.split("\t")
                    if len(line_list)==2:
                        words.append(line_list[0])
                        if set_type!='test':
                            tags.append(line_list[1])
                        else:
                            tags.append("O")
                else:
                    lines.append((len(lines), words, tags))
                    words = []
                    tags = []
            return lines

    @classmethod
    def _read_csv(cls, input_file, set_type):
        """Reads a tab separated value file."""
        raise NotImplementedError

    @classmethod
    def _read_json(self, data_path, set_type):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
        lines = []
        for d in data:
            id_ = d["id"]
            text = d["text"]
            entities = d["entities"]
            new_entities = []
            for ent in entities:
                new_entities.append(Entity(id=str(ent["id"]),
                                           start=ent["start"],
                                           end=ent["end"],
                                           mention=ent["mention"],
                                           type=ent["type"]))
            lines.append((id_, text, new_entities))
        return lines


    def _create_examples(self, lines, set_type)->List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for i, text, entities  in lines:
            # id
            id_ = i
            guid = "%s-%s" % (set_type, id_)

            examples.append(
                InputExample(
                    id=guid,
                    text=text,
                    entities=entities,
                )
            )
        return examples

    def get_examples(self, path, mode, data_type):
        """
        Args:
            mode: train, dev, test
        """

        logger.info("LOOKING AT {}".format(mode))
        if data_type == "conll":
            data_path = os.path.join(path, "{}.txt".format(mode))
            examples = self._create_examples(lines=self._read_file(data_path, set_type=mode), set_type=mode)
        elif data_type == "json":
            data_path = os.path.join(path, "{}.json".format(mode))
            examples = self._create_examples(lines=self._read_json(data_path, set_type=mode), set_type=mode)
        elif data_type == "ann":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return examples

processors = {
    'bertmlp': BertMlpProcessor,
}

def convert_examples_to_features(examples:List[InputExample],
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
            text, input_ids, offset_mapping, token_span = ent2token_spans(text, entities, tokenizer)
            labels = ["O"] * len(input_ids)
            for token_start_index, token_end_index, token_type in token_span:
                labels[token_start_index] = "B-" + token_type
                for tk_id in range(token_start_index+1, token_end_index+1):
                    labels[tk_id] = "I-"+ token_type

        elif isinstance(text, list):
            if args.token_subword:
                text, tokens, labels, offset_mapping = list_subword_token(text, entities, tokenizer, label_map, unk_token)
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
        offset_mapping = offset_mapping +[0]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        input_ids = [tokenizer.cls_token_id] + input_ids
        labels = [pad_token_label_id] + labels
        token_type_ids = [cls_token_segment_id] + token_type_ids
        offset_mapping =  [0] + offset_mapping
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
        example.entities = labels
        setattr(example, "offsets_mapping", offset_mapping)


    return features, examples

def list_subword_token(text, entities, tokenizer, label_map, unk_token):
    offset_mapping = []
    tokens = []

    slot_labels = []
    # 转为 sub-token 级别 数据中有一些 输入 不是单词级别，比如日期，需要做转化
    label_id2label = {v: k for k, v in label_map.items()}

    for index, (word, slot_label) in enumerate(zip(text, entities)):
        words_ = tokenizer.tokenize(word)
        if not words_:
            text.remove(word)
            continue

        tokens.extend(words_)
        slot_label_nonstart = slot_label.replace("B-", "I-")
        slot_labels.extend([slot_label] + [slot_label_nonstart] * (len(words_) - 1))
        offset_mapping.append(len(words_))

    return text, tokens, slot_labels, offset_mapping

def cut_long_text(examples:List[InputExample],
             max_seq_length:int,
             tokenizer,
             special_tokens_count:int=2,
             is_chinese=False):

    if not (len(examples) and isinstance(examples[0].text, str)):
        raise NotImplementedError

    new_examples = []
    process = PreProcessBase(tokenizer=tokenizer,
                             max_length=max_seq_length,
                             special_tokens_count=special_tokens_count,
                             is_chinese=is_chinese)

    for index, example in enumerate(examples):
        new_examples+=process.cut_text(example)

    return new_examples

def ent2token_spans(text:Text, entities:List[Entity], tokenizer):
    """

    :param text: 原始文本
    :param entity_list: [{'id': 1, 'start': 0, 'end': 2, 'mention': 'EU', 'type': 'ORG'}]
    :param tokenizer: 分词器
    :return:
    """
    ent2token_spans = []
    inputs = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    '''
    {'input_ids': [6224, 7481, 2218, 6206, 6432, 10114, 8701, 9719, 8457, 8758], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'offset_mapping': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (9, 14), (15, 18), (18, 21), (21, 24)]}
    '''
    token2char_span_mapping = inputs["offset_mapping"]#每个切分后的token在原始的text中的起始位置和结束位置
    text2token = tokenizer.tokenize(text, add_special_tokens=False)
    #['见', '面', '就', '要', '说', 'say', 'hello', 'yes', '##ter', '##day']
    for ent_span in entities:
        ent = text[ent_span.start:ent_span.end]
        ent2token = tokenizer.tokenize(ent, add_special_tokens=False)

        # 然后将按字符个数标注的位置  修订 成 分完词 以token为个体的位置
        token_start_indexs = [i for i, v in enumerate(text2token) if v==ent2token[0]]
        token_end_indexs = [i for i, v in enumerate(text2token) if v==ent2token[-1]]

        # 分词后的位置 转为字符寻址 要和之前标的地址要一致 否则 就出错了
        token_start_index = list(filter(lambda x:token2char_span_mapping[x][0]==ent_span.start, token_start_indexs))
        token_end_index = list(filter(lambda x:token2char_span_mapping[x][-1] == ent_span.end, token_end_indexs))

        if len(token_start_index)==0 or len(token_end_index)==0:
            continue
            # 无法对应的token_span中
        token_span = (token_start_index[0],token_end_index[0], ent_span.type)
        ent2token_spans.append(token_span)
    return text, inputs["input_ids"], token2char_span_mapping, ent2token_spans

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
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode,args.data_type)
        elif mode == "dev":
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode, args.data_type)
        elif mode == "test":
            examples = processor.get_examples(os.path.join(args.data_dir, args.data_name, args.data_type), mode, args.data_type)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        if args.local_rank in [-1, 0]:
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)

    if args.is_cut_text:
        examples = cut_long_text(examples=examples,
                                 max_seq_length=args.max_seq_length,
                                 tokenizer=tokenizer,
                                 special_tokens_count=2,
                                 is_chinese=False)

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