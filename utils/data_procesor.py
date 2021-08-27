import configparser
import numpy as np
import jieba
from transformers import TFBertModel, BertTokenizer, BertConfig

import tensorflow as tf
from tensorflow import optimizers

classes = {'negative': 0, 'positive': 1}
class_list = [name for name, index in classes.items()]


class DataProcessor:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.word_token_file = config.get('text_classification', 'word_token_file')
        # train_data_file = config.get('text_classification', 'train_data_file')
        # self.train_dataset, self.val_dataset = self.load_data_file(train_data_file)
        self.word2token, self.token2word = {}, {}
        self.load_vocab()
        self.max_sequence_length = config.getint('text_classification', 'max_sequence_length')
        self.PADDING = config.get('text_classification', 'PADDING')
        self.UNKNOWN = config.get('text_classification', 'UNKNOWN')
        bert_config = BertConfig.from_json_file('./bert-base-chinese/config.json')

        self.bert_model = TFBertModel.from_pretrained('bert-base-chinese', config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', config=bert_config)

    def load_vocab(self, sentences=None):
        with open(self.word_token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word, word_token = row.split('\t')[0], int(row.split('\t')[1])
                self.word2token[word] = word_token
                self.token2word[word_token] = word
        return

    def sentence_to_vector(self, sentence):
        vector = []
        cut_words = jieba.cut(str(sentence).strip())
        sentence = list(cut_words)
        sentence = self.padding(sentence)
        print(sentence)

        for word in sentence:
            if word in self.word2token:
                vector.append(self.word2token[word])
            else:
                vector.append(self.word2token[self.UNKNOWN])
        return np.array([vector], dtype=int)

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence
