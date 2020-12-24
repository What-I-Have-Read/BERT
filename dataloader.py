# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataloader.py
@time: 2020/12/23 23:46:03

这一行开始写关于本文件的说明与解释


'''
import numpy as np
from typing import List
import tensorflow as tf
from transformers import BertTokenizer


class Tokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def tokenize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        return tokenized_text

    def encode_plus(self, text, max_length):
        tokenized = self.tokenizer.encode_plus(
            text, max_length=max_length, padding="max_length")
        input_ids = tf.convert_to_tensor(tokenized['input_ids'])
        attention_mask = tf.convert_to_tensor(tokenized['attention_mask'])
        token_type_ids = tf.convert_to_tensor(tokenized['token_type_ids'])
        return input_ids, attention_mask, token_type_ids

    def batch_tokenize(self, texts: List[str]):
        return [self.tokenize(text) for text in texts]


class DataLoader():
    def __init__(self, instances, batch_size=64, tokenizer=Tokenizer(),
                 label_list=["complete", "incomplete"], vocab=None):
        self.instances = instances
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.label_list = label_list

        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.dataset = self._gen_dataset()

    def __len__(self):
        return len(self.instances)

    def _data_generator(self):
        for instance in self.instances:
            input_ids, attention_mask, token_type_ids = self.tokenizer.encode_plus(
                instance["tokens"], max_length=32)
            label = self.label_map[instance["label"]]
            yield {"input_ids": input_ids, "attention_mask": attention_mask,
                   "token_type_ids": token_type_ids}, label

    @property
    def dummy_inputs(self):
        """
        取训练所用dataset第一个元素的x值作为dummy_inputs来初始化模型结构，需要train与predict使用seq_length相同
        """
        return list(self.dataset.take(1))[0][0]

    def _gen_dataset(self):
        dataset = tf.data.Dataset.from_generator(self._data_generator,
                                                 output_types=({"input_ids": tf.int32, "attention_mask": tf.int32,
                                                                "token_type_ids": tf.int32}, tf.int32))
        return dataset

    def __iter__(self):
        dataset = self.dataset.batch(
            batch_size=self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(1)
        for x, y in dataset:
            yield x, y
