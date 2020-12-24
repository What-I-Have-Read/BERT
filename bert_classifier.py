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
        # todo
        # tf.keras.model.save_model之后再load
        # 在预测与构造模型结构使用seq_length不同的seq_length的tokens时候会出现error:
        ''' 
            (.eg: model(dummy_inputs, training=False) 构建模型, 这里dummy_inputs seq_length=6, 
             预测时候则需要输入tokens的seq_length=32)
            (解决方案：1. 改用save_weights, 2. 直接使用32长度的dummy_inputs；
             针对方案2，我们使用dataloader中的dummy_inputs)
        >>>
            Could not find matching function to call loaded from the SavedModel. Got:
            Positional arguments (2 total):
                * {'input_ids': <tf.Tensor 'inputs:0' shape=(1, 32) dtype=int32>}
                * False
            Keyword arguments: {}

            Expected these arguments to match one of the following 4 option(s):

            Option 1:
            Positional arguments (2 total):
                * {'input_ids': TensorSpec(shape=(None, 6),...
        >>>
        '''
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
