import configparser
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import jieba
from engines.models.textrnn import TextRNN
from engines.models.textcnn import TextCNN

from tensorflow.keras import optimizers, metrics
import datetime
from utils.metrics import calculate_metrics
from transformers import TFBertModel, BertTokenizer, BertConfig

import os

classes = {'negative': 0, 'positive': 1}
class_list = [name for name, index in classes.items()]


class TrainProcessor:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.word_token_file = config.get('text_classification', 'word_token_file')
        self.model_type = config.get('text_classification', 'text_classifier')
        if config.get('text_classification', 'optimizers') == 'Adam':
            self.optimizers = optimizers.Adam(lr=0.001)
        else:
            self.optimizers = optimizers.RMSprop(lr=0.001)
        self.epochs = config.getint('text_classification', 'epochs')
        self.batch_size = config.getint('text_classification', 'batch_size')
        self.max_sequence_length = config.getint('text_classification', 'max_sequence_length')
        self.PADDING = config.get('text_classification', 'PADDING')
        self.UNKNOWN = config.get('text_classification', 'UNKNOWN')
        self.is_early_stop = config.getboolean('text_classification', 'is_early_stop')
        self.patient = config.getint('text_classification', 'patient')
        self.checkpoints_dir = config.get('text_classification', 'checkpoints_dir')
        self.max_to_keep = config.getint('text_classification', 'max_to_keep')
        self.checkpoint_name = config.get('text_classification', 'checkpoint_name')
        self.stop_words = config.get('text_classification', 'stop_words')
        self.embedding_method = config.get('text_classification', 'embedding_method')
        train_data_file = config.get('text_classification', 'train_data_file')
        test_data_file = config.get('text_classification', 'test_data_file')
        bert_config = BertConfig.from_json_file('./bert-base-chinese/config.json')

        self.bert_model = TFBertModel.from_pretrained('bert-base-chinese', config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', config=bert_config)

        self.use_bert = True
        if self.use_bert:
            self.X, self.y, self.att_mask = self.load_train_file(train_data_file, use_bert=True)
        else:
            self.train_dataset, self.val_dataset = self.load_train_file(train_data_file)
            self.test_dataset = self.load_test_file(test_data_file)

    def set_dataset(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):

        # meter
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # summary
        log_dir = 'logs/' + self.model_type + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        acc_meter = metrics.Accuracy()
        loss_meter = metrics.Mean()

        #
        step_count = 0
        best_f1score = 0.0
        patient_count = 0

        if self.model_type == 'text_rnn':
            model = TextRNN()
        elif self.model_type == 'text_cnn':
            model = TextCNN()

        # if self.embedding_method == 'Bert':
        #     from transformers import TFBertModel
        #     bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        # else:
        #     bert_model = None
        # load checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=self.checkpoints_dir,
                                                        max_to_keep=self.max_to_keep,
                                                        checkpoint_name=self.checkpoint_name)
        # checkpoint.restore(checkpoint_manager.latest_checkpoint)

        for epoch in range(self.epochs):
            self.train_dataset, self.val_dataset = self.to_dataset(self.X, self.y, self.att_mask)
            # 训练集
            for step, batch_dataset in tqdm(self.train_dataset.batch(self.batch_size).enumerate(),
                                            desc='epoch:' + str(epoch)):
                step_count += 1
                if self.use_bert:
                    X, y, att_mask = batch_dataset
                    X, _ = self.bert_model(X, attention_mask=att_mask)
                else:
                    X, y = batch_dataset

                with tf.GradientTape() as tape:
                    logits = model(X, training=1)
                    y_onehot = tf.one_hot(y, depth=len(classes))
                    loss = tf.metrics.categorical_crossentropy(y_true=y_onehot, y_pred=logits, from_logits=True)
                    loss_mean = tf.reduce_mean(loss, axis=0)
                    loss_meter.update_state(loss_mean)
                # 反向传播
                grads = tape.gradient(loss_mean, model.trainable_variables)
                self.optimizers.apply_gradients(zip(grads, model.trainable_variables))
                # print('model.trainable_variables:',model.trainable_variables)
                # 记录loss
                if step_count % 100 == 0:
                    loss_cross_view = float(loss_meter.result().numpy())
                    with summary_writer.as_default():
                        tf.summary.scalar('train-loss', float(loss_cross_view), step=step_count)
                    loss_meter.reset_states()

            # 验证集
            y_true, y_pred = np.array([]), np.array([])
            for step, batch_dataset in self.val_dataset.batch(self.batch_size).enumerate():
                X, y, att_mask = batch_dataset
                X, _ = self.bert_model(X, attention_mask=att_mask)
                logits = model(X, training=0)
                prob = tf.nn.softmax(logits, axis=1)
                pred = tf.argmax(prob, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)
                acc_meter.update_state(y, pred)
                y_true = np.append(y_true, y)
                y_pred = np.append(y_pred, pred)

            # 计算metrics
            all_metrics = calculate_metrics(y_true, y_pred)
            acc = float(acc_meter.result().numpy())
            with summary_writer.as_default():
                tf.summary.scalar('val-acc', acc, step=epoch)
            acc_meter.reset_states()
            print("epoch %d: f1score:%f precision:%f recall:%f acc:%f" % (epoch, all_metrics['f1_score'],
                                                                          all_metrics['precision_score'],
                                                                          all_metrics['recall_score'], acc))
            if all_metrics['f1_score'] > best_f1score:
                best_f1score = all_metrics['f1_score']
                checkpoint_manager.save()
                patient_count = 0
                best_epoch = epoch
                best_all_metrics = all_metrics
                best_acc = acc
                print(patient_count)
            else:
                patient_count += 1
                print(patient_count)

            if self.is_early_stop and patient_count >= self.patient:
                print('early stop , in epoch %d , best f1score:%f precision:%f recall:%f acc:%f ' % (
                    best_epoch, best_all_metrics['f1_score'],
                    best_all_metrics['precision_score'],
                    best_all_metrics['recall_score'], best_acc))
                return

        return

    def load_train_file(self, data_file, use_bert=False):
        classes = {}
        X = []
        y = []
        att_mask = []
        if use_bert:
            train_raw_data = pd.read_csv('data/online_shopping.csv', encoding='utf-8').sample(frac=1)
            # cat,label,sentence
            for sentence, label in zip(train_raw_data['sentence'], train_raw_data['label']):
                try:
                    sentence_ids = self.tokenizer.encode(sentence)
                except:
                    continue

                if len(sentence_ids) >= self.max_sequence_length:
                    sentence_ids = sentence_ids[0:self.max_sequence_length]
                    X.append(sentence_ids)
                    y.append(int(label))
                    att_mask.append([1] * self.max_sequence_length)
                else:
                    sentence_ids = self.tokenizer.encode(sentence)
                    sentence_ids += [0] * (self.max_sequence_length - len(sentence_ids))
                    X.append(sentence_ids)
                    y.append(int(label))
                    att_mask.append([1] * len(sentence_ids) + [0] * (self.max_sequence_length - len(sentence_ids)))
            return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32), np.array(att_mask, dtype=np.int32)
        else:
            train_raw_data = pd.read_csv(data_file, encoding='utf-8').sample(frac=1)
            train_raw_data, val_raw_data = train_raw_data[:int(len(train_raw_data) * 0.9)], train_raw_data[
                                                                                            int(len(
                                                                                                train_raw_data) * 0.9):]
            train_dataset = self.get_dataset(train_raw_data)
            val_dataset = self.get_dataset(val_raw_data)

            return train_dataset, val_dataset

    def load_test_file(self, data_file):
        train_raw_data = pd.read_csv(data_file, encoding='utf-8').sample(frac=1)
        test_dataset = self.get_dataset(train_raw_data)
        return test_dataset

    def to_dataset(self, X, y, att_mask):
        print(X.shape)
        sample_num = len(X)
        index = np.arange(sample_num)
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
        att_mask = att_mask[index]
        X_train = X[0:int(0.9 * sample_num)]
        y_train = y[0:int(0.9 * sample_num)]
        att_mask_train = att_mask[0:int(0.9 * sample_num)]
        X_val = X[int(0.9 * sample_num):]
        y_val = y[int(0.9 * sample_num):]
        att_mask_val = att_mask[int(0.9 * sample_num):]
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val))
        return train_dataset, val_dataset

    def get_dataset(self, raw_data, use_bert=False):
        X = []
        y = []
        raw_data = raw_data.loc[raw_data.label.isin(class_list)]
        raw_data.label = raw_data.label.map(lambda x: classes[x])
        word2token, token2word = self.load_vocab(raw_data['sentence'])

        X, y = self.prepare_data(raw_data['sentence'], raw_data['label'], word2token)
        y = tf.cast(y, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(10000).batch(self.batch_size)
        return dataset

    def load_vocab(self, sentences=None):
        word2token, token2word = {}, {}
        with open(self.word_token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word, word_token = row.split('\t')[0], int(row.split('\t')[1])
                word2token[word] = word_token
                token2word[word_token] = word
        return word2token, token2word

    def prepare_data(self, sentences, labels, word2token):
        """
        输出X矩阵和y向量
        """
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.remove_stopwords(record[0], self.stop_words)
            sentence = self.padding(sentence)
            # label = tf.one_hot(record[1], depth=2)
            label = record[1]
            tokens = []
            for word in sentence:
                if word in word2token:
                    tokens.append(word2token[word])
                else:
                    tokens.append(word2token[self.UNKNOWN])
            X.append(tokens)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def remove_stopwords(self, item, stopwords):

        cut_item = jieba.cut(str(item).strip())
        if stopwords:
            item = [word for word in item if word not in stopwords and word != ' ']
        else:
            item = list(cut_item)
            item = [word for word in item if word != ' ']
        return item

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

    def get_stop_words(self):
        stop_words_list = []
        try:
            with open(self.stop_words, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list
