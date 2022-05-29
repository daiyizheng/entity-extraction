#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 23:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : build_mrc_data.py
from __future__ import annotations

import json, os
import collections
data_dir = "../../../../datasets/conll"
label_file = "mrc_labels.json"
labels2default = json.load(open(os.path.join(data_dir, label_file), 'r', encoding="utf-8"))["default"]
labels = json.load(open(os.path.join(data_dir, label_file), 'r', encoding="utf-8"))["labels"]

for file in ["train_mrc.json", "dev_mrc.json", "test_mrc.json"]:
    input_path = os.path.join(data_dir, file)
    file_type = file.split("_")[0]
    output_path = os.path.join(data_dir, file.split("_")[0]+".json")
    contents = json.load(open(input_path, "r", encoding="utf-8"))
    new_contents = []
    for item in contents:
        if file_type!="test":
            entities = item["entities"]
            enk = collections.defaultdict(list)
            for ent in entities:
                enk[ent['type']].append(ent)
            for k in enk:
                new_contents.append({"id":item["id"], "text":item["text"], "ner_cate":k, "query":labels2default[k], "entities":enk[k]})
        else:
            for la in labels:
                new_contents.append({"id": item["id"], "text": item["text"], "ner_cate": la, "query": labels2default[la],"entities": item["entities"]})

    json.dump(new_contents, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

