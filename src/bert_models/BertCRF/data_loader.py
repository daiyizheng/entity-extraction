#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 15:21
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : data_loader.py
from __future__ import annotations

import copy
import json
import os, sys
import logging
from typing import List, Dict, Tuple, Union, Text

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.bert_models.BertCRF.utils import subword_label_alignment

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

class CutExample(object):
    def __init__(self, id, text, entities=None,  sub_id=None, cut_text=None, cut_entities=None,
                 cut_number=None, cut_len=None,labels=[], pre_labels=[], offsets_mapping=None):
        self.id = id
        self.text = text
        self.entities=entities
        self.sub_id = sub_id
        self.cut_text = cut_text
        self.cut_entities = cut_entities
        self.cut_number = cut_number
        self.cut_len = cut_len
        self.labels = labels
        self.pre_labels = pre_labels
        self.offsets_mapping = offsets_mapping



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


class BertCRFProcessor(object):
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
                    lines.append((words, tags))
                    words = []
                    tags = []
            return lines

    @classmethod
    def _read_csv(cls, input_file, set_type):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file)
        lines = []
        for index, item in tqdm(df.iterrows()):
            if set_type=="test":
                lines.append([index, item["text1"], item["text2"], 0])
            else:
                lines.append([index, item["text1"], item["text2"], item['label']])
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

    def get_examples(self,path, mode, max_length, tokenizer, special_tokens_count=2):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
        logger.info("LOOKING AT {}".format(path))
        examples = self._create_examples(lines=self._read_file(data_path, set_type=mode), set_type=mode)
        cut_examples = self.cut_long_text(examples, max_length = max_length, tokenizer=tokenizer,
                                          special_tokens_count=special_tokens_count)
        return cut_examples

    def _tokenize_chinese_chars(self, token: Text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in token:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def concat_word(self, new_token: Text, cur_token: Text):
        if cur_token.strip() == "":
            return new_token
        if new_token == "":
            new_token += cur_token
        elif self._is_chinese_char(ord(new_token[-1])) and self._is_chinese_char(ord(cur_token[0])):  # 前中文，当前中文
            new_token += cur_token
        elif self._is_chinese_char(ord(new_token[-1])) and not self._is_chinese_char(ord(cur_token[0])):  # 前中文，当前英文
            new_token += cur_token
        elif not self._is_chinese_char(ord(new_token[-1])) and self._is_chinese_char(ord(cur_token[0])):  # 前英文，当前中文
            new_token += cur_token
        else:
            new_token = new_token + " " + cur_token
        return new_token

    def concat_words(self, token: Text):
        """
        :param token:
        :return:
        """
        new_token = ""
        token = self._tokenize_chinese_chars(token)  # 你 (hello) a
        token_list = token.strip().split(" ")
        for cur_token in token_list:
            new_token = self.concat_word(new_token, cur_token)
        return new_token

    def conll_to_text(self, token_list:List, label_list:List)->Union:
        """
        conll转为text文本标签
        :param token_list: ["Role", "of", "number"]
        :param label_list: ["O","O", "O"]
        :return: {"text":"Role of number",entities:[{"id":"T1", "start":10, "end":12, "mention":"of"，"type":"GENE"}]}
        """
        text = ""
        entities = []
        index = 0
        id_num= 0
        while index<len(token_list):
            token = token_list[index].strip()
            token = self.concat_words(token) # 内部拼接
            text = self.concat_word(text, token)
            if label_list[index][0] == "B":
                label_type = label_list[index].split("-")[-1]
                start_ids = len(text) - len(token)
                mention = token
                end_ids = len(text)
                while(index+1<len(token_list) and label_list[index+1][0]=="I"):
                    index += 1
                    token = token_list[index].strip()
                    token = self.concat_words(token)
                    text = self.concat_word(text, token)
                    mention = mention+ " " + token
                    end_ids = len(text)
                index += 1
                id_num += 1
                entities.append({"id":id_num, "start":start_ids, "end":end_ids, "mention":mention, "type":label_type})
            else:
                index += 1
        return text, entities

    def cut_long_text(self, examples:List[InputExample], max_length, tokenizer, **kwargs):
        cut_examples = []
        for (ex_index, example) in enumerate(examples):
            id = example.guid
            tokens = example.words
            labels = example.labels
            if isinstance(tokens, list):
                text, entities = self.conll_to_text(tokens, labels)
                exp = {"id":id, "text":text, "entities":entities} #  {"text":"Role of number",entities:[{"id":"T1", "start":10, "end":12, "mention":"of"，"type":"GENE"}]}
                train_samples = self.cut_text(exp, max_length, tokenizer, **kwargs)

            elif isinstance(tokens, str):
                text = tokens
                entities= labels
                exp = {"id": id, "text": text, "entities": entities}
                train_samples = self.cut_text(exp, max_length, tokenizer, **kwargs)
            else:
                raise ValueError

            cut_examples += [CutExample(
                id=id,
                text=text,
                entities=entities,
                sub_id=item["sub_id"],
                cut_text=item["cut_text"],
                cut_entities=item["cut_entities"],
                cut_number=item["cut_number"],
                cut_len=item["cut_len"],
                labels=[],
                pre_labels=[],
                offsets_mapping=None
            ) for item in train_samples]
        return cut_examples



    def cut_text(self, example: Dict, max_length, tokenizer, **kwargs) -> List[Dict]:
        """
        :param example:{"id":"101246", "text": "Role of", "task_name":"ner", "entities":["id":0, "start":10, "end":12, "mention":"of"，"type":"GENE"]}
        :param max_length: default=512
        :return:
        {'id': '101246', 'task_name':"ner",'sub_id': 1,
          'text': 'Role of phosphatidylinositol 3-kinase-Akt pathway',
          'entities':[{'type': 'Gene', 'id': '0', 'mention': 'NPM', 'end': 7, 'start': 4}]
          'cut_len': 170
          'cut_entities': [{'type': 'Gene', 'id': '0', 'mention': 'NPM', 'end': 7, 'start': 4}],
          'cut_text': 'The NPM/ALK fusion gene, formed by the t(2;5)'}
        """
        train_samples = []

        text = example['text']
        entities = example['entities']
        cut_sents_list = [(0, text)]
        ## 一阶截断
        if self.is_ge_max_len(cut_sents_list, max_length, tokenizer, **kwargs):
            cut_sents_list = self.cut_sentences_v1((0, text), punctuation_special_char={"."})  # [(0, "1111"), (65, "222")]
        ## 二阶截断
        if self.is_ge_max_len(cut_sents_list, max_length, tokenizer, **kwargs):
            two_cut_sents_list = cut_sents_list
            cut_sents_list = []
            for cut_sent in two_cut_sents_list:
                if len(tokenizer(cut_sent[1])["input_ids"]) > max_length:
                    cut_sents_list += self.cut_sentences_v2([cut_sent], entities)
                else:
                    cut_sents_list += [cut_sent]

        ## 截断后纠正标签的起始和结束的位置
        # {'id': 0, 'text': 'KRAS G12V mutation', 'entities': [{"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}, {"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}]}
        correct_cut_sents_list = self.correct_tag_position_v1(cut_sents_list, entities)
        ## 校验标签是否对齐
        self.check_ner_postion_v1(correct_cut_sents_list)
        for idx, ccsl in enumerate(correct_cut_sents_list):
            train_samples.append(
                {
                    "id": example["id"],
                    "text": example["text"],
                    "entities": example["entities"],
                    "sub_id": idx,
                    "cut_text": ccsl["text"],
                    "cut_entities": ccsl["entities"],
                    "cut_number": len(correct_cut_sents_list),
                    "cut_len": len(ccsl['text'])
                }
            )

        return train_samples

    def is_ge_max_len(self, sents_list: List[Tuple], max_len: int, tokenizer, **kwargs) -> bool:
        special_tokens_count = kwargs.get("special_tokens_count", 2)
        for cut_sent in sents_list:
            if len(tokenizer(cut_sent[1])["input_ids"]) > max_len-special_tokens_count:
                return True
        return False

    def correct_tag_position_v1(self, all_cut_sents: List[Tuple], annotaion: List[Dict]):
        """
        [{"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}]
        return [{'id': 0, 'text': 'KRAS G12V mutation', 'entities': [{"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}, {"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}]}]
        """
        all_data = []
        sent_ids = 0
        for c_s in all_cut_sents:
            cut_text = c_s[1]
            cut_start = c_s[0]
            cut_end = cut_start + len(cut_text)
            entities = []
            for an in annotaion:
                e_start = an['start']
                e_end = an['end']
                if cut_start <= e_start < cut_end and cut_start <= e_end < cut_end:
                    entities.append({"id": an["id"], "start": e_start - cut_start, "end": e_end - cut_start,
                                     "mention": an['mention'], "type": an['type']})
            all_data.append({"id": sent_ids, "text": cut_text, "entities": entities})
            sent_ids += 1
        return all_data

    def check_ner_postion_v1(self, correct_cut_sent_list: List[Dict]) -> None:
        """
        :param correct_cut_sent_list:
           [{'id': 0, 'text': 'KRAS G12V mutation',
            'entities': [{"id":0, "start":12, "end": 13, "mention"："of", "type":"gene"}]}]
        :return:
        """
        for one_cut_sent in correct_cut_sent_list:
            cut_text = one_cut_sent["text"]
            for ent in one_cut_sent['entities']:
                mention = ent["mention"]
                cut_start = ent["start"]
                cut_end = ent["end"]
                if cut_text[cut_start:cut_end] != mention:
                    logger.warning("entity is not aligned")
                    logger.warning("cut_text:%s" % cut_text)
                    logger.warning("cut_start:%s" % cut_start)
                    logger.warning("cut_end:%s" % cut_end)
                    logger.warning("mention:%s" % mention)
                    logger.warning("cut tag:%s" % cut_text[cut_start:cut_end])
                    raise ValueError

    def cut_sentences_v1(self, sent_tupe: Tuple, punctuation_special_char={".", ";"}) -> List[Tuple]:
        """
        [(0, "1111"), (65, "222")], tupe第一位是原文的开始位置，第二位是原文截断后的位置
        """
        init_start = sent_tupe[0]
        sent = sent_tupe[1]
        sents = []
        string = ""
        start = 0
        for index, char in enumerate(sent):
            if char == " " and index - 1 > 0 and (sent[index - 1] in punctuation_special_char):
                string += sent[index]
                sents.append((init_start + start, string))
                string = ""
                start = index + 1
            else:
                string += sent[index]
        if string:
            sents.append((init_start + start, string))
        return sents

    def cut_sentences_v2(self, sent_tupes: List[Tuple], entities: Dict) -> List[Tuple]:
        new_sent_tupes = []
        for sent_tupe in sent_tupes:
            init_start = sent_tupe[0]
            c_text = sent_tupe[1]
            mid = len(c_text) // 2
            entities = sorted(entities, key=lambda x: x["start"])
            while True:
                flag = False
                for ent in entities:
                    if ent["start"] <= mid < ent["end"]:
                        flag = True

                if not flag or mid >= len(c_text):
                    break
                mid += 1
            new_sent_tupes += [(init_start, c_text[:mid]), (init_start + mid, c_text[mid:])]
        return new_sent_tupes






processors = {
    'bertcrf': BertCRFProcessor,
}

def convert_examples_to_features(examples:List[CutExample],
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

        # Tokenize
        # token = tokenizer.tokenize(example.words)
        text = example.cut_text
        entities = example.cut_entities

        text, input_ids, offset_mapping, labels = subword_label_alignment(text, entities, tokenizer)
        tokens = [tokenizer.convert_ids_to_tokens(tk) for tk in input_ids]

        labels = [label_map[l] for l in labels]

        # print(tokens, " ".join(example.words), )

        # Account for [CLS] and [SEP]
        if len(input_ids) > (max_seq_length - special_tokens_count):
            input_ids = input_ids[:(max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            offset_mapping = offset_mapping[:(max_seq_length - special_tokens_count)]


        # Add [SEP] token
        tokens = tokens + [sep_token]
        input_ids = input_ids + [tokenizer.sep_token_id]
        labels  = labels +[pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * (len(input_ids))

        # Add [CLS] token
        tokens = [cls_token] + tokens
        input_ids = [tokenizer.cls_token_id] + input_ids
        labels =  [pad_token_label_id] + labels
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
        example.offsets_mapping = offset_mapping
        example.labels = labels

    return features, examples


def load_and_cache_examples(args, tokenizer, mode):
    pad_token_label_id = args.ignore_index or args.label_map["PAD"]
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[args.model_type](args)

    # # level 标签的频次
    # label2freq = json.load(
    #     open(args.label2freq_level_dir, "r", encoding="utf-8"),
    # )

    # 加载label list
    # label_list_level = get_labels(args.label_file_level_dir)

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
        cut_examples = torch.load(cached_examples_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(args.data_dir, mode, args.max_seq_length,
                                              tokenizer, special_tokens_count=2)
        elif mode == "dev":
            examples = processor.get_examples(args.data_dir, mode, args.max_seq_length,
                                              tokenizer, special_tokens_count=2)
        elif mode == "test":
            examples = processor.get_examples(args.data_dir, mode, args.max_seq_length,
                                              tokenizer, special_tokens_count=2)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features, cut_examples = convert_examples_to_features(
            examples,
            args.max_seq_length,
            args.label_map,
            tokenizer,
            pad_token_label_id=pad_token_label_id,
            special_tokens_count=2)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(cut_examples, cached_examples_file)


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

    return dataset, cut_examples