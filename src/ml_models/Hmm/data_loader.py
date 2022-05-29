#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 9:50
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : data_loader.py
from __future__ import annotations
from typing import List, Dict, Tuple, Union, Text
import copy
import json
import os,sys
import logging

from tqdm import tqdm
import pandas as pd
import torch

from src.ml_models.Hmm.utils import spanlabel2seqlabel
from src.ml_models.Hmm.utils import ent2token_spans

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
                entities.append(self.postion_to_string(self.pre_labels, entitie, self.offsets_mapping, self.text, ids))
                ids += 1
                entitie = [index]
            elif tag[0] == "I":
                entitie.append(index)
            else:
                if entitie:
                    entities.append(self.postion_to_string(self.pre_labels, entitie, self.offsets_mapping, self.text, ids))
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
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

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
                 text,
                 tokens,
                 labels,
                 offset_mapping):
        self.text = text
        self.tokens = tokens
        self.labels = labels
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


class HMMProcessor(object):
    """Processor for the BERT data set """

    def __init__(self, args):
        self.args = args


    def _read_conll_file(self, input_file, set_type=None):
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
                    lines.append((words, tags))
                    words = []
                    tags = []
            return lines

    def _read_csv(self, input_file, set_type):
        """Reads a tab separated value file."""
        raise  NotImplementedError

    def _read_json(self, input_file, set_type):
        """
        [{"text":"xxx", entities:[]}]
        :param input_file:
        :param set_type:
        :return:
        """
        data = json.load(open(input_file, 'r', encoding="utf-8"))
        lines = []
        for d in data:
            lines.append((d["text"], d['entities']))
        return lines

    def _create_examples(self, lines, set_type)->List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (word, label) in enumerate(lines):
            # id
            id_ = i
            guid = "%s-%s" % (set_type, id_)

            examples.append(
                InputExample(
                    guid=guid,
                    words=word,
                    labels=label,
                )
            )
        return examples

    def get_examples(self,path, mode, max_length, tokenizer, data_type, special_tokens_count=2):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
        logger.info("LOOKING AT {}".format(path))
        if data_type == "conll":
            examples = self._create_examples(lines=self._read_conll_file(data_path, set_type=mode), set_type=mode)
        elif data_type == "json_list":
            raise NotImplementedError
        elif data_type == "json_text":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return examples

processors = {
    'hmm': HMMProcessor,
}

def convert_examples_to_features(examples:List[InputExample],
                                 max_seq_length,
                                 label_map,
                                 tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize
        # token = tokenizer.tokenize(example.words)
        text = example.words
        entities = example.labels
        if isinstance(text, str):
            # text: xx xx xx  entities:[{id:"", start:"", end:"", type:"", metion:""}]
            tokens, offset_mapping, labels_span = ent2token_spans(text=text,
                                                            entity_list=entities,
                                                            tokenizer=tokenizer)
            labels = spanlabel2seqlabel(span_entity=labels_span, length=len(tokens))

        elif isinstance(text, list):
            tokens = text
            labels = entities
            offset_mapping = [(i, i+1)for i in range(len(tokens))]
        else:
            raise NotImplementedError

        if len(tokens) > (max_seq_length):
            labels = labels[:max_seq_length]
            tokens = tokens[:max_seq_length]
            offset_mapping = offset_mapping[:max_seq_length]

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("slot_labels: %s" % " ".join([str(x) for x in labels]))

        features.append(InputFeatures(text=text,
                      tokens=tokens,
                      labels=labels,
                      offset_mapping=offset_mapping))

    return features, None

def load_and_cache_examples(args, tokenizer, mode):

    processor = processors[args.model_type](args)

    # # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cache_{}_{}_{}".format(
            args.task_name,
            args.model_type,
            mode
        )
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              tokenizer,
                                              args.data_type,
                                              special_tokens_count=0)
        elif mode == "dev":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              tokenizer,
                                              args.data_type,
                                              special_tokens_count=0)
        elif mode == "test":
            examples = processor.get_examples(args.data_dir,
                                              mode,
                                              args.max_seq_length,
                                              tokenizer,
                                              args.data_type,
                                              special_tokens_count=0)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features, _ = convert_examples_to_features(
            examples,
            args.max_seq_length,
            args.label_map,
            tokenizer)


        #  由于slot_labels_ids的矩阵在globalponter模型中比较大，在保存加载比较缓慢，建议注释掉
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    tokens = [f.tokens for f in features]
    labels = [f.labels for f in features]

    return [tokens, labels], features
