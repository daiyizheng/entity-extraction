#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 21:59
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : data_loader.py
from __future__ import annotations

import copy
import json
import os
import logging
from typing import List, Dict, Tuple, Union, Text

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.bert_models.BertMRC.utils import ent2token_spans, ent2char_list, entlist2entspan, entspan2entlist

logger = logging.getLogger(__file__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        sentence1: list. The words of the sequence.
        sentence2: list. The words of the sequence.
        label: (Optional) string. The label of the example.
    """
    def __init__(self, guid, text, query, entities, ner_cate, offset_mapping=None):
        self.guid = guid
        self.text = text
        self.query = query
        self.entities = entities
        self.ner_cate = ner_cate
        self.offset_mapping = offset_mapping

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
                 start_position_ids,
                 end_position_ids,
                 ner_cate_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position_ids = start_position_ids
        self.end_position_ids = end_position_ids,
        self.ner_cate_ids = ner_cate_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertMRCProcessor(object):
    """Processor for the BERT data set """

    def __init__(self, args):
        self.args = args

    def _create_examples(self, lines, set_type):
        examples = []
        for index, line in enumerate(lines):#[{id:"",text:"", "entities":[], "query":"", ner_cate:""}]
            examples.append(
                InputExample(
                    guid=index,
                    text=line["text"],
                    entities=line["entities"],
                    query=line["query"],
                    ner_cate=line["ner_cate"]
                )
            )
        return examples

    def _read_json(self, data_path, set_type):
        contents = json.load(open(data_path, "r", encoding="utf-8"))
        return contents

    def _read_txt(self, data_path, set_type):
        raise NotImplementedError

    def get_examples(self,path, mode, data_type):
        """
        Args:
            mode: train, dev, test
        """

        logger.info("LOOKING AT {}".format(path))
        if data_type == "conll":
            data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
            examples = self._create_examples(lines=self._read_txt(data_path, set_type=mode), set_type=mode)
        elif data_type == "json":
            data_path = os.path.join(self.args.data_dir, "{}.json".format(mode))
            examples = self._create_examples(lines=self._read_json(data_path, set_type=mode), set_type=mode)
        else:
            raise NotImplementedError

        return examples

processors = {
    'bertmrc': BertMRCProcessor,
}

def convert_examples_to_features(examples:List[InputExample],
                                 max_seq_length,
                                 label_map,
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

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text = example.text
        query = example.query
        entities = example.entities
        ner_cate = example.ner_cate
        if isinstance(text, str):
            text, input_ids, offset_mapping, label_span = ent2token_spans(text=text, entity_list=entities, tokenizer=tokenizer)
            # label_span [{id:"", start:"", end:"", type:"", metion:""},{}] labels 是一个开区间的，已经减去-1比如[0, 0]
            tokens = [tokenizer.convert_ids_to_tokens(tk) for tk in input_ids]
            labels = label_span

        elif isinstance(text, List):
            _, input_ids, offset_mapping, label_list = ent2char_list(text, entities, tokenizer)
            tokens=text
            labels = entlist2entspan(text, label_list)
        else:
            raise NotImplementedError

        query_inputs = tokenizer(query, add_special_tokens=False)
        query_inputs_ids = query_inputs["input_ids"]
        query_tokens = [tokenizer.convert_ids_to_tokens(tk) for tk in query_inputs_ids]
        max_tokens_for_doc = max_seq_length - len(query_inputs_ids)

        # 设置闭区间-1, end位置-1，开区间
        labels_seq = [(label["start"], label["end"], label["type"]) for label in labels]

        # print(tokens, " ".join(example.words), )
        label_add_list = []
        if len(input_ids) > (max_tokens_for_doc - special_tokens_count):
            tokens = tokens[:(max_tokens_for_doc - special_tokens_count)]
            input_ids = input_ids[:(max_tokens_for_doc - special_tokens_count)]
            offset_mapping = offset_mapping[:(max_tokens_for_doc - special_tokens_count)]
            for id, label in enumerate(labels_seq):
                label_add = []
                label_oringin_start = label[0]
                label_oringin_end = label[1]
                if max_tokens_for_doc - special_tokens_count <= label_oringin_start:  # 删除
                    continue
                elif max_tokens_for_doc - special_tokens_count == label_oringin_start + 1:  # 边界值
                    label_add.append(max_tokens_for_doc - special_tokens_count - 1)
                    label_add.append(max_tokens_for_doc - special_tokens_count - 1)
                    label_add.append(label[2])
                    label_add_list.append(label_add)
                elif label_oringin_end + 1 <= max_tokens_for_doc - special_tokens_count:
                    label_add_list.append(list(label))
                else:
                    label_add.append(label[0])
                    label_add.append(max_tokens_for_doc - special_tokens_count - 1)
                    label_add.append(label[2])
                    label_add_list.append(label_add)
        else:
            label_add_list = [list(label).copy() for label in labels_seq]

        # Add [SEP] token
        tokens = tokens + [sep_token]
        input_ids = input_ids + [tokenizer.sep_token_id]
        # labels  = labels +[pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * (len(input_ids))
        offset_mapping.append((0,0))

        # Add [CLS] token
        tokens = [cls_token] + tokens
        input_ids = [tokenizer.cls_token_id] + input_ids
        # labels =  [pad_token_label_id] + labels
        token_type_ids = [cls_token_segment_id] + token_type_ids
        offset_mapping.insert(0, (0, 0))
        for ele in label_add_list:
            ele[0] += 1
            ele[1] += 1

        ## 添加quer_token
        tokens = tokens + query_tokens
        input_ids = input_ids + query_inputs_ids
        token_type_ids = token_type_ids + [1] * (len(query_inputs_ids))
        offset_mapping = offset_mapping+ [(0, 0)]*len(query_inputs_ids)

        ## Add [SEP]
        tokens = tokens + [sep_token]
        input_ids = input_ids + [tokenizer.sep_token_id]
        token_type_ids = token_type_ids + [1]
        offset_mapping = offset_mapping +[(0, 0)]

        start_position_ids = [0] * len(tokens)
        end_position_ids = [0] * len(tokens)

        for lb in label_add_list:
            start_position_ids[lb[0]] = 1
            end_position_ids[lb[1]] = 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        start_position_ids = start_position_ids + ([pad_token_label_id] * padding_length)
        end_position_ids = end_position_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_length)
        assert len(start_position_ids) == max_seq_length, "Error with start_position_ids length {} vs {}".format(len(labels), max_seq_length)
        assert len(end_position_ids) == max_seq_length, "Error with end_position_ids length {} vs {}".format(len(labels), max_seq_length)
        ner_cate_ids = label_map[ner_cate]
        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("start_position_ids: %s" % " ".join([str(x) for x in start_position_ids]))
            print("end_position_ids: %s" % " ".join([str(x) for x in end_position_ids]))
            print("label_seqs: %s", json.dumps(label_add_list, ensure_ascii=False))
            print("ner_cate_ids: %s" %ner_cate_ids)

        features.append(
            InputFeatures(
                input_ids= input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                start_position_ids=start_position_ids,
                end_position_ids=end_position_ids,
                ner_cate_ids= ner_cate_ids
            )
        )
        example.offset_mapping = offset_mapping

    return features, examples


def load_and_cache_examples(args, tokenizer, mode):
    pad_token_label_id = args.pad_token_label_id or args.label_map.get("[PAD]") or -100
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[args.model_type](args)

    # # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cache_{}_{}_{}_{}".format(
            args.model_type,
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)
        )
    )

    cached_examples_file = os.path.join(
        args.data_dir,
        "cache_examples_{}_{}_{}_{}".format(
            args.model_type,
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)
        )
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = torch.load(cached_examples_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              )
        elif mode == "dev":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              )
        elif mode == "test":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              )
        else:
            raise Exception("For mode, Only train, dev, test is available")

        if args.is_cut_text:
            raise NotImplementedError

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features, examples = convert_examples_to_features(
            examples,
            args.max_seq_length,
            args.label_map,
            tokenizer,
            pad_token_label_id=pad_token_label_id,
            special_tokens_count=3)

        if args.local_rank in [-1, 0]:
         # 由于slot_labels_ids的矩阵在globalponter模型中比较大，在保存加载比较缓慢，建议注释掉
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(examples, cached_examples_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    start_position_ids = torch.tensor([f.start_position_ids for f in features], dtype=torch.long)
    end_position_ids = torch.tensor([f.end_position_ids for f in features], dtype=torch.long)
    ner_cate_ids = torch.tensor([f.ner_cate_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        start_position_ids,
        end_position_ids,
        ner_cate_ids
    )

    return dataset, examples