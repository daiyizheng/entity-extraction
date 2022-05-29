#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 22:03
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : conll_to_json.py
from __future__ import annotations

import codecs
import json
import os
from typing import Text, List, Union



def _tokenize_chinese_chars(token: Text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in token:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def _is_chinese_char(cp):
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

def concat_word(new_token: Text, cur_token: Text):
    if cur_token.strip() == "":
        return new_token
    if new_token == "":
        new_token += cur_token
    elif _is_chinese_char(ord(new_token[-1])) and _is_chinese_char(ord(cur_token[0])):  # 前中文，当前中文
        new_token += cur_token
    elif _is_chinese_char(ord(new_token[-1])) and not _is_chinese_char(ord(cur_token[0])):  # 前中文，当前英文
        new_token += cur_token
    elif not _is_chinese_char(ord(new_token[-1])) and _is_chinese_char(ord(cur_token[0])):  # 前英文，当前中文
        new_token += cur_token
    else:
        new_token = new_token + " " + cur_token
    return new_token

def concat_words(token: Text):
    """
    :param token:
    :return:
    """
    new_token = ""
    token = _tokenize_chinese_chars(token)  # 你 (hello) a
    token_list = token.strip().split(" ")
    for cur_token in token_list:
        new_token = concat_word(new_token, cur_token)
    return new_token
def conll_to_text(token_list:List, label_list:List)->Union:
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
        token = concat_words(token) # 内部拼接
        text = concat_word(text, token)
        if label_list[index][0] == "B":
            label_type = label_list[index].split("-")[-1]
            start_ids = len(text) - len(token)
            mention = token
            end_ids = len(text)
            while(index+1<len(token_list) and label_list[index+1][0]=="I"):
                index += 1
                token = token_list[index].strip()
                token = concat_words(token)
                text = concat_word(text, token)
                mention = mention+ " " + token
                end_ids = len(text)
            index += 1
            id_num += 1
            entities.append({"id":id_num, "start":start_ids, "end":end_ids, "mention":mention, "type":label_type})
        else:
            index += 1
    return text, entities

data_dir = "../../../../datasets/conll"
for file in ["train.txt", "dev.txt", "test.txt"]:
    file_type = file.split(".")[0]
    inpyt_path = os.path.join(data_dir, file)
    output_path = os.path.join(data_dir, file.split(".")[0]+"_mrc.json")
    with open(inpyt_path, "r", encoding="utf-8") as f:
        sentences_list, tags_list = [], []

        text_list = []
        tag_list = []

        for line in f:
            line = line.strip()
            if len(line) > 0:
                text, tag = line.split("\t")
                text_list.append(text)
                if file_type == "test":
                    tag_list.append("O")
                else:
                    tag_list.append(tag)
            else:
                if len(tag_list):
                    sentences_list.append(text_list)
                    tags_list.append(tag_list)
                text_list = []
                tag_list = []

    contents = []
    for index in range(len(sentences_list)):
        tokens = sentences_list[index]
        labels = tags_list[index]
        text, entities = conll_to_text(tokens, labels)
        contents.append({"id":index, "text":text, "entities":entities})

    file_json = codecs.open(output_path, "w", encoding="utf-8")
    file_json.write(json.dumps(contents, indent=2, ensure_ascii=False))
    file_json.close()





