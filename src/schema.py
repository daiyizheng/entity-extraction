#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/21 21:36
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : schema.py

from __future__ import annotations

from typing import Optional, Union, Text, List, Dict, Tuple, Any
from collections.abc import Iterable
import dataclasses

@dataclasses.dataclass
class Entity(object):
    mention: str
    type: str
    start: int
    end: int
    id: Optional[str] = None

@dataclasses.dataclass
class Relation(object):
    type: str
    arg1: Entity
    arg2: Entity
    id: Optional[str] = None

@dataclasses.dataclass
class Event(object):
    id: Optional[str] = None

@dataclasses.dataclass
class Example(object):
    text: Union[str, Iterable[str]]
    entities: list[Entity] = dataclasses.field(default_factory=list)
    relations: list[Relation] = dataclasses.field(default_factory=list)
    events: list[Event] = dataclasses.field(default_factory=list)
    id: Optional[str] = None
    task_name:Optional[str] = None