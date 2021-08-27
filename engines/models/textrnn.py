from abc import ABC

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import configparser


class TextRNN(keras.Model, ABC):

    def __init__(self):
        super(TextRNN, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.use_bert = True
        self.embedding_dim = config.getint('text_classification', 'embedding_dim')
        self.sequence_length = config.getint('text_classification', 'max_sequence_length')
        self.voc_size = config.getint('text_classification', 'voc_size')  # 词典的大小，传入进去之后有何作用？
        self.state_dim = config.getint('text_classification', 'state_dim')  # 状态向量的维度
        self.embedding = layers.Embedding(self.voc_size + 1, self.embedding_dim, input_length=self.sequence_length,
                                          mask_zero=True)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(self.state_dim, return_sequences=True, activation='relu', use_bias=True), merge_mode='sum')
        self.dense = layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                  bias_regularizer=tf.keras.regularizers.l2(0.1))
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dropout = layers.Dropout(rate=0.5)
        # self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layerNorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.layerNorm3 = tf.keras.layers.LayerNormalization(axis=-1)

        # self.attention_Wq = tf.keras.layers.Dense(300)
        # self.attention_Wk = tf.keras.layers.Dense(300)
        # self.attention_Wv = tf.keras.layers.Dense(300)
        # self.alpha=0
        # self.attention_output=0

    # @tf.function
    def call(self, inputs, training=None):
        """

        :param inputs: [b,sequence_length]
        :param training:
        :return:
        """
        x = inputs
        if not self.use_bert:
            x = self.embedding(inputs)  # [b,sequence_length,embedding_dim]
        # batch_norm_out = self.batch_norm(x, training=training)
        h = self.bilstm(x)  # [b,sequence_length,state_dim]
        # attention_inputs = tf.split(tf.reshape(h, [-1, 32]), self.sequence_length, axis=0)
        # # attention
        # attention_size = 300
        # # with tf.name_scope('attention'), tf.variable_scope('attention'):
        # # 定义W_w
        # attention_w = tf.Variable(tf.random.truncated_normal([self.state_dim, attention_size], stddev=0.1),
        #                           name='attention_w')  # W_w
        # # 定义b_w
        # attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')  # b_w
        # u_list = []
        # for t in range(self.sequence_length):
        #     # 公式1
        #     u_t = tf.tanh(tf.matmul(attention_inputs[t], attention_w) + attention_b)  #
        #     u_list.append(u_t)
        # # 定义u_w
        # u_w = tf.Variable(tf.random.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
        # attn_z = []
        # for t in range(self.sequence_length):
        #     # 公式2括号里面的内容
        #     z_t = tf.matmul(u_list[t], u_w)
        #     attn_z.append(z_t)
        # # transform to batch_size * sequence_length
        # attn_zconcat = tf.concat(attn_z, axis=1)
        # # 运用softmax函数
        # alpha = tf.nn.softmax(attn_zconcat)
        # # transform to sequence_length * batch_size * 1 , same rank as outputs
        # alpha_trans = tf.reshape(alpha,[-1,self.sequence_length, 1])
        # # print('alpha_trans', alpha_trans)
        # # 公式3
        # attention_output = h * alpha_trans

        dropout_outputs = self.dropout(h, training=training)
        # dropout_outputs = self.layerNorm3(dropout_outputs)
        flatten_output = self.flatten(dropout_outputs)
        logits = self.dense(flatten_output)
        return logits
