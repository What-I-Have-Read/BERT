# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: bert_classifier.py
@time: 2020/12/23 23:42:55

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf

from transformers import TFBertModel, BertTokenizer


class BertClassifier(tf.keras.Model):
    def __init__(self, bert_type, num_classes):
        super(BertClassifier, self).__init__()
        # self.bert = TFBertModel.from_pretrained(bert_type)
        self.bert_layer = TFBertModel.from_pretrained(bert_type).bert
        # 这里我们需要使用base_model.bert; 否则使用tf.keras load_model会出错：
        # 解决方案参考issue: https://github.com/huggingface/transformers/issues/3627
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(num_classes,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                               stddev=0.02),
                                           name="classifier")

    @property
    def dummy_inputs(self):
        return {"input_ids": tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])}

    @tf.function()
    def call(self, inputs, training=True):
        # [last_hidden_state: (b, s, 768), pooler_output: (b, 768)]
        bert_output = self.bert_layer(**inputs)
        # bert_cls_hidden_state = bert_output[0][:, 0, :]
        # bert main_layer中的pooler层为：取的[cls]对应的hidden_state算的
        pooled_output = bert_output[1]
        # (使用的dense层，activation=tanh, units=config.hidden_size, 与classification需要的[cls]向量相同)

        output = self.dropout(pooled_output, training=training)
        output = self.dense(output)
        return output


if __name__ == "__main__":
    bert_classifier = BertClassifier(
        bert_type="bert-base-chinese", num_classes=2)
    bert_classifier(bert_classifier.dummy_inputs)
    bert_classifier.summary()
