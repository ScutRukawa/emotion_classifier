from abc import ABC

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class TextRNN(keras.Model, ABC):

    def __init__(self):
        super(TextRNN, self).__init__()
        self.embedding_dim = 150  # 词嵌入维度
        self.sequence_length = 200  # 句子长度
        self.voc_size = 6411  # 词典的大小，传入进去之后有何作用？
        self.state_dim = 32  # 状态向量的维度
        # self.model = keras.models.Sequential()
        self.embedding = layers.Embedding(self.voc_size, self.embedding_dim, input_length=self.sequence_length)
        # self.flatten = self.model.add(layers.Flatten())  应用场景?
        self.bilstm = layers.Bidirectional(layers.LSTM(self.state_dim, return_sequences=True))
        # LSTM 有多少个cell？ 和输入序列的长度有何关系？ 根据句子长度？
        self.dense = layers.Dense(2, activation='softmax')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dropout= layers.Dropout(rate=0.5)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [b,sequence_length]
        :param tranning:
        :return:
        """

        # x = tf.reshape(inputs, [-1, self.sequence_length * self.embedding_dim])
        x = self.embedding(inputs)  # [b,sequence_length,embedding_dim]
        h = self.bilstm(x)
        flatten_output = self.flatten(h)
        y = self.dense(flatten_output)
        return y
