from abc import ABC
import configparser

import tensorflow as tf


class TextCNN(tf.keras.Model, ABC):
    def __init__(self):
        super(TextCNN, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        filter_nums = config.getint('text_classification', 'filter_nums')
        max_sequence_length = config.getint('text_classification', 'max_sequence_length')
        voc_size = config.getint('text_classification', 'voc_size')
        embedding_dim = config.getint('text_classification', 'embedding_dim')

        self.embedding = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=embedding_dim,
                                                   input_length=max_sequence_length)

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=filter_nums, kernel_size=2, padding='valid')
        self.pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=max_sequence_length - 2 + 1, padding='valid')

        self.conv2d_2 = tf.keras.layers.Conv2D(filters=filter_nums, kernel_size=3, padding='valid')
        self.pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=max_sequence_length - 3 + 1, padding='valid')

        self.conv2d_3 = tf.keras.layers.Conv2D(filters=filter_nums, kernel_size=4, padding='valid')
        self.pooling_3 = tf.keras.layers.MaxPooling2D(pool_size=max_sequence_length - 4 + 1, padding='valid')

        self.dropout = tf.keras.layers.Dropout(0.5, name='dropout')
        self.dense = tf.keras.layers.Dense(2,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    def call(self, inputs, training=None, mask=None):
        inputs = self.embedding(inputs)

        inputs = tf.expand_dims(inputs, -1)

        pooled_output = []
        conv_out1 = self.conv2d_1(inputs)
        pooling_out1 = self.pooling_1(conv_out1)
        pooled_output.append(pooling_out1)

        conv_out2 = self.conv2d_2(inputs)
        pooling_out2 = self.pooling_2(conv_out2)
        pooled_output.append(pooling_out2)

        conv_out3 = self.conv2d_3(inputs)
        pooling_out3 = self.pooling_3(conv_out3)
        pooled_output.append(pooling_out3)

        concat_out = tf.keras.layers.concatenate(pooled_output)
        flatten_outputs = self.flatten(concat_out)
        dropout_outputs = self.dropout(flatten_outputs, training)
        outputs = self.dense(dropout_outputs)

        return outputs
