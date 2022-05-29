#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 9:14
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : build_vocab.py

import os, codecs
import json

from tqdm import tqdm

tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}

def check_label(front_label, follow_label):
    """
    TODO:
    :param front_label
    :param follow_label
    """
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
            front_label.endswith(follow_label.split("-")[1]) and \
            front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False

class pre_processsing(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tag_type = set()
        self.vocab = set()

    def txt2json(self):
        """
        txt2json函数
        """
        ## train
        sentences_list, tags_list = self._read_txt(os.path.join(self.data_dir, "train.txt"))
        self._save_json(sentences_list, tags_list, os.path.join(self.data_dir, "train.json"))
        ## dev
        sentences_list, tags_list = self._read_txt(os.path.join(self.data_dir, "dev.txt"))
        self._save_json(sentences_list, tags_list, os.path.join(self.data_dir, "dev.json"))
        ## test
        sentences_list, tags_list = self._read_txt(os.path.join(self.data_dir, "test.txt"))
        self._save_json(sentences_list, tags_list, os.path.join(self.data_dir, "test.json"))
        ## 保存tags类型
        self._save_tags_type(os.path.join(self.data_dir, "tagging.txt"))
        ## 构建词典vocab
        self._build_vocab(os.path.join(self.data_dir, "vocab.txt"))


    def _save_json(self, sentences_list, tags_list, path):
        """
        保存json格式
        :param sentences_list:list 句子数组
        :param tags_list:list 标签数组
        :param path:str 保存路径
        """
        assert len(sentences_list) == len(tags_list), "the lens of tag_lists is not eq to word_lists"
        """
        ## json数据格式 注意 在英文中，位置的不一致性，因为英文原始词是空格隔开的，位置的起始是不一致的
        [{"id":1,"text":"", "text_token":[], "labels":[O, B-OR, I-OR], "ann_tags";[[]], "candidate_entities":[]}
        ]
        """
        file_type = os.path.basename(path).split(".")[0]
        file_json = codecs.open(path, "w", encoding="utf-8")
        contents = []
        idx = 0
        for index, tags in tqdm(enumerate(tags_list)):
            text = " ".join(sentences_list[index])
            text_token = sentences_list[index]
            ## 将text_token 作为创建vocab的预料
            self.vocab.update(text_token)
            labels = tags
            ann_tags = []
            candidate_entities = []
            if file_type !="test":
                ann_tags, candidate_entities= self._build_ann_tags(text_token, labels)
            ## 保存标签类型
            if file_type!="test":
                for tag in tags:
                    self.tag_type.add(tag)
            contents.append({"id":idx, "text":text, "text_token":text_token, "labels":labels, "ann_tags":ann_tags, "candidate_entities":candidate_entities})
            idx +=1

        file_json.write(json.dumps(contents, indent=2, ensure_ascii=False))
        file_json.close()

    def _build_vocab(self, path):
        """
        :param path:str 词表保存路径
        """
        vocab = list(self.vocab)
        fw = codecs.open(path, "w", encoding="utf-8")
        for t in vocab:
            fw.write(t)
            fw.write("\n")
        ## 词表补充特殊字符
        fw.write(unk_token)
        fw.write("\n")
        fw.write(pad_token)
        fw.write("\n")
        fw.close()


    def _save_tags_type(self, path):
        """
        :param path:str 保存路径
        """
        tag_type = list(self.tag_type)
        with codecs.open(path, "w", encoding="utf-8") as fw:
            for t in tag_type:
                fw.write(t)
                fw.write("\n")

    def _build_ann_tags(self, chars, tags):
        """
         BIO表标签提取实体，及实体位置，其他标签请转换为BIO在提取， 目前只支持中文
        :param chars:list 句子单词数组
        :param labels:list 标签数组
        """
        entities = []
        entity = []
        for index, (char, tag) in enumerate(zip(chars, tags)):
            entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
            if not entity_continue and entity:
                entities.append(entity)
                entity = []
            entity.append([index, char, tag, entity_continue])
        if entity:
            entities.append(entity)
        ann_tags = []
        candidate_entities = []
        T = 1
        for entity in entities:
            if entity[0][2].startswith("B-"):
                candidate_entities.append("".join(chars[entity[0][0]:entity[-1][0]+1]))
                ann_tags.append(["T"+str(T), entity[0][2].split("-")[1], entity[0][0], entity[-1][0] + 1, "".join(chars[entity[0][0]:entity[-1][0]+1])])
                T += 1
        return ann_tags, candidate_entities

    def _read_txt(self, path):
        """
        :param input_file:
        :return: list [text, category, intention, slot_label]
        """
        sentences_list, tags_list = [], []
        with codecs.open(path, "r", encoding="utf-8") as f:
            text_list = []
            tag_list = []
            file_type = os.path.basename(path).split(".")[0]
            for line in f:
                line = line.strip()
                if len(line)>0:
                    text, tag = line.split("\t")
                    text_list.append(text)
                    if file_type == "test":
                        tag_list.append(None)
                    else:
                        tag_list.append(tag)
                else:
                    if len(tag_list):
                        sentences_list.append(text_list)
                        tags_list.append(tag_list)
                    text_list = []
                    tag_list = []
            return sentences_list, tags_list

    def json2ann(self, file_name):
        """
        将文件存储为brat工具可视化
        :param file_name:str 文件名称
        """
        with codecs.open(os.path.join(self.data_dir, file_name), "w", encoding="utf-8") as fr:
            line = fr.read()
            contents = json.loads(line)
        ## 创建ann目录
        ann_path = os.path.join(self.data_dir,"ann")
        if not os.path.exists(ann_path):
            os.mkdir(ann_path)

        ##遍历
        for index, item in tqdm(enumerate(contents)):
            txt_cont = codecs.open(os.path.join(ann_path, str(index)+".txt"), "w", encoding="utf-8")
            txt_ann = codecs.open(os.path.join(ann_path, str(index) + ".ann"), "w", encoding="utf-8")
            text = item["text"]
            txt_cont.write(text)
            txt_cont.close()
            anns = item["ann_tags"]
            for tag in anns:
                txt_ann.write(tag[0]+"\t"+tag[1]+" "+str(tag[2])+" "+str(tag[3])+"\t"+tag[4])
                txt_ann.write("\n")
            txt_ann.close()

    def json2txt(self):
        ##train
        tran_sample = self._read_json(os.path.join(self.data_dir, "train.json"))
        self._save_txt(tran_sample, os.path.join(self.data_dir, "train.txt"))
        ## dev
        dev_sample = self._read_json(os.path.join(self.data_dir, "dev.json"))
        self._save_txt(dev_sample, os.path.join(self.data_dir, "dev.txt"))
        ## test
        test_sample = self._read_json(os.path.join(self.data_dir, "test.json"))
        self._save_txt(test_sample, os.path.join(self.data_dir, "test.txt"))
        ## 保存tags类型
        self._save_tags_type(os.path.join(self.data_dir, "tagging.txt"))


    def _save_txt(self, contents, path):
        """
         json数据保存为txt数据
        :param contents:json josn数据
        :param path:str 数据保存路径
        """
        with codecs.open(path, "w", encoding="utf-8") as fw:
            for item in tqdm(contents):
                text_token = item["text_token"]
                labels = item["labels"]
                assert len(text_token) == len(labels), "the lens of text_token is not eq to labels"
                for index, tag in enumerate(labels):
                    fw.write(text_token[index]+"\t"+tag)
                    fw.write("\n")
                fw.write("\n")

    def _read_json(self, path):
        """
        读取json文件
        :param path:str 文件路径
        """
        with codecs.open(path, "r", encoding="utf-8") as fr:
            contents = json.loads(fr.read())
        return contents

if __name__ == '__main__':
    unk_token= "[UNK]"
    pad_token = "[PAD]"
    pre = pre_processsing(data_dir=r"../../../../datasets/conll")
    pre.txt2json()


