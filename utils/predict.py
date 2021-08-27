import jieba
from utils.data_procesor import DataProcessor
import configparser
from engines.models.textrnn import TextRNN
import tensorflow as tf
import os

classes = {'negative': 0, 'positive': 1}
class_list = [name for name, index in classes.items()]


class Predictor:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.checkpoint_name = config.get('text_classification', 'checkpoint_name')
        self.checkpoints_dir = config.get('text_classification', 'checkpoints_dir')
        self.max_to_keep = config.getint('text_classification', 'max_to_keep')
        model_type = config.get('text_classification', 'text_classifier')
        if model_type == 'text_rnn':
            self.model = TextRNN()

        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=self.checkpoints_dir,
                                                        max_to_keep=self.max_to_keep,
                                                        checkpoint_name=self.checkpoint_name)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("loading model %s" % model_type)

    def predict(self, sentence):
        vector = self.data_processor.sentence_to_vector(sentence)
        logits = self.model(vector, training=0)
        prob = tf.nn.softmax(logits, axis=1)
        print('prob:', prob)
        predict = tf.argmax(prob, axis=1)
        predict = predict.numpy()[0]
        return class_list[predict]
