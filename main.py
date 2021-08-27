import jieba
from utils.data_procesor import DataProcessor
from utils.predict import Predictor
from utils.train import TrainProcessor
import tensorflow as tf
import numpy as np
import configparser
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizer, BertConfig
import pandas as pd

#
if __name__ == '__main__':
    trainProcessor = TrainProcessor()
    trainProcessor.train()

    # y=1
    # print(tf.one_hot(y,depth=2))
