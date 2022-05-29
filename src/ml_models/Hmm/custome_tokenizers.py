#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 0:00
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : tokenizers.py
from __future__ import annotations
from typing import Text, List, Tuple, Dict

from nltk import wordpunct_tokenize, word_tokenize

def doubleQuotes(text):
    """
    双引号改成单引号
    :param text:
    :return:
    """
    return text.replace('"', "'")

class WordTokenizerFast():
    def __init__(self, *args, **kwargs):
        self.do_lower_case = kwargs.pop("do_lower_case", True)
        self.word2id = kwargs.pop("word2id", {})
        self.id2word = { self.word2id[k]: k for k in self.word2id}
        self.unk_token = kwargs.pop("unk_token", "[UNK]")
        self.pad_token = kwargs.pop("pad_token", "[PAD]")
    def _check_unk(self):
        if self.unk_token not in self.word2id:
            raise ValueError("unk_token not in word2id")

    def tokenize(self, text:Text)->List:
        text = text.strip()
        text = doubleQuotes(text)
        if self.do_lower_case:
            text = text.lower()
        return word_tokenize(text)

    def vocab_size(self):
        return len(self.word2id)

    def __call__(self, text:Text)->Dict:
        self._check_unk()
        word_list = self.tokenize(text)
        offset_mapping = self.create_offset_mapping(text, word_list)
        input_ids = [self.convert_tokens_to_ids(w) for w in word_list]
        return {"input_ids":input_ids, "offset_mapping":offset_mapping}

    def convert_ids_to_tokens(self, token_id)->Text:
        return self.id2word.get(token_id, self.unk_token)

    def convert_tokens_to_ids(self, token):
        return self.word2id.get(token, self.word2id[self.unk_token])

    def create_offset_mapping(self, text:Text, word_list:List)->List[Tuple[int, int]]:
        idx = 0
        offset_mapping = []
        start = 0
        while idx < len(word_list):
            while text[start] == " ":
                start += 1
            end = start + len(word_list[idx])
            assert text[start:end] == word_list[idx], "单词切分错误！:%s" % text
            offset_mapping.append((start, end))
            start = end
            idx += 1
        return offset_mapping

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class WordPunctTokenizerFast:

    def __init__(self, *args, **kwargs):
        self.do_lower_case = kwargs.pop("do_lower_case", True)
        self.word2id = kwargs.pop("word2id", {})
        self.unk_token = kwargs.pop("unk_token", "[UNK]")
        self.pad_token = kwargs.pop("pad_token", "[PAD]")


    def _check_unk(self):
        if self.unk_token not in self.word2id:
            raise ValueError("unk_token not in word2id")

    def tokenize(self, text:Text)->List:
        if self.do_lower_case:
            text = text.lower()
        return wordpunct_tokenize(text)

    def vocab_size(self):
        return len(self.word2id)

    def __call__(self, text:Text)-> Dict:
        self._check_unk()
        if self.do_lower_case:
            text = text.lower()
        text = text.strip()
        word_list = wordpunct_tokenize(text)
        idx = 0
        offset_mapping = []
        start = 0
        while idx < len(word_list):
            while text[start] == " ":
                start += 1
            end = start + len(word_list[idx])
            assert text[start:end] == word_list[idx], "单词切分错误！"
            offset_mapping.append((start, end))
            start = end
            idx += 1
        input_ids = [self.word2id.get(w, self.word2id[self.unk_token]) for w in word_list]
        return {"input_ids":input_ids, "offset_mapping":offset_mapping}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)



# if __name__ == '__main__':
#     text = "The enantioselective binding of [Fe(4,7-dmp)(3)](2+) (dmp: 4,7-dimethyl-1,10-phenantroline) and [Fe(3,4,7,8-tmp)(3)](2+) (tmp: 3,4,7,8-tetramethyl-1,10-phenanthroline) to calf-thymus DNA (ct-DNA) has been systematically studied by monitoring the circular dichroism (CD) spectral profile of the iron(II) complexes in the absence and presence of ct-DNA."
#     tokenizer = WordPunctTokenizerFast()
#     print(tokenizer.tokenize(text))
