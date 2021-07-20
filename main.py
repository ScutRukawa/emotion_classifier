from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, SimpleRNN, RNN, Dropout, Bidirectional
from tensorflow.keras import optimizers
import tensorflow as tf
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import numpy as np
import os
from engines.models.textrnn import TextRNN

embedding_dim_train = 150
import pandas as pd

max_sequence_length = 200
PADDING = 'PAD'
UNKNOWN = 'UNK'
word_token2id, id2word_token = {}, {}
batch_size = 64


def load_vocab(sentences=None):
    with open('./data/token2id', 'r', encoding='utf-8') as infile:
        for row in infile:
            row = row.strip()
            word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
            word_token2id[word_token] = word_token_id
            id2word_token[word_token_id] = word_token
    # vocabsize = len(word_token2id)
    # print(word_token2id)
    # print(id2word_token)
    return word_token2id, id2word_token


def prepare_data(sentences, labels, token2id):
    """
    输出X矩阵和y向量
    """
    X, y = [], []
    for record in tqdm(zip(sentences, labels)):
        sentence = remove_stopwords(record[0], ' ')
        sentence = padding(sentence)
        label = tf.one_hot(record[1], depth=2)
        tokens = []
        for word in sentence:
            if word in word_token2id:
                tokens.append(word_token2id[word])
            else:
                tokens.append(word_token2id[UNKNOWN])
        X.append(tokens)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_dataset(df):
    classes.items()
    df = df.loc[df.label.isin(class_list)]
    df.label = df.label.map(lambda x: classes[x])
    # X, y = prepare_w2v_data(df.sentence, df.label)
    token2id, id2token = load_vocab(df['sentence'])
    X, y = prepare_data(df['sentence'], df['label'], id2word_token)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    return dataset


def remove_stopwords(item, stopwords):
    import jieba
    cut_item = jieba.cut(str(item).strip())
    if stopwords:
        item = [word for word in item if word not in stopwords and word != ' ']
    else:
        item = list(cut_item)
        item = [word for word in item if word != ' ']
    return item


def padding(sentence):
    """
    长度不足max_sequence_length则补齐
    :param sentence:
    :return:
    """
    if len(sentence) < max_sequence_length:
        sentence += [PADDING for _ in range(max_sequence_length - len(sentence))]
    else:
        sentence = sentence[:max_sequence_length]
    return sentence


# def prepare_w2v_data(sentences, labels):
#     X, y = [], []
#     for record in tqdm(zip(sentences, labels)):
#         # print(count)
#         # do something
#         sentence = remove_stopwords(record[0], '')
#         sentence = padding(sentence)
#         label = tf.one_hot(record[1], depth=2)
#         vector = []
#         for word in sentence:
#             if word in w2v_model.wv.vocab:
#                 vector.append(w2v_model[word].tolist())
#                 # print(w2v_model[word].tolist())
#             else:
#                 vector.append([0] * embedding_dim_train)
#         X.append(vector)
#         y.append(label)
#     print('complete word to vector')
#     X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
#     return X, y


#
if __name__ == '__main__':
    # model = Model()
    # model.build(input_shape=(64, 500))
    # model.summary()
    train_df = pd.read_csv('./data/game_comments.csv', encoding='utf-8').sample(frac=1)
    train_df, dev_df = train_df[:int(len(train_df) * 0.9)], train_df[int(len(train_df) * 0.9):]
    test_df = pd.read_csv('./data/dev_data.csv', encoding='utf-8').sample(frac=1)

    classes = {'negative': 0, 'positive': 1}
    class_list = [name for name, index in classes.items()]
    # class_index=[index for name, index in classes.items()]

    # load word2vector model
    # w2v_model = Word2Vec.load('./model/w2v_model/w2v_model.pkl')
    # embedding_dim = w2v_model.vector_size
    # print(w2v_model)
    # vocab_size = len(w2v_model.wv.vocab)
    word_token2id, id2word_token = load_vocab()

    # X_train, y_train = prepare_data(train_df.sentence, train_df.label, word_token2id)
    # X_test, y_test = prepare_data(dev_df.sentence, dev_df.label, word_token2id)
    dataset_train = get_dataset(train_df)
    train_iter = iter(dataset_train)
    sample = next(train_iter)

    dataset_val = get_dataset(dev_df)
    dataset_test = get_dataset(test_df)

    state_dim = 32  # 64效果不行
    epochs = 10
    model = TextRNN()
    model.build(input_shape=(batch_size,max_sequence_length))
    model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(dataset_train, epochs=epochs, batch_size=batch_size,
                        validation_data=dataset_val, validation_steps=2)
    # model.summary()
    loss_and_acc = model.evaluate(dataset_test)
    print('loss=' + str(loss_and_acc[0]))
    print('acc=' + str(loss_and_acc[1]))


