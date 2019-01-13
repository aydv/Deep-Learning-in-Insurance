import os
import numpy as np
from keras import (initializers, optimizers, regularizers)
from keras.layers import (BatchNormalization, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, Input, MaxPooling1D, PReLU,
                          SeparableConv1D, SpatialDropout1D, add)
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import keras
from nltk.corpus import stopwords
import string
import nltk

class DPCNN:
    def __init__(self):
        self.le = LabelEncoder()
        self.tk = Tokenizer()
        self.maxlen = 80
        self.embed_size = 100
        self.filter_nr = 32
        self.filter_size = 3
        self.max_pool_size = 3
        self.max_pool_strides = 2
        self.dense_nr = 256
        self.spatial_dropout = 0.2
        self.dense_dropout = 0.5
        self.batch_size = 200
        self.epochs = 15

    def __get_model(self, max_features, sequence_len,
                    num_classes):
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)
        comment = Input(shape=(sequence_len, ))
        emb_comment = Embedding(max_features, self.embed_size)(comment)
        emb_comment = SpatialDropout1D(self.spatial_dropout)(emb_comment)

        block1 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(emb_comment)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        resize_emb = SeparableConv1D(
            self.filter_nr,
            kernel_size=1,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(emb_comment)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        block1_output = MaxPooling1D(
            pool_size=self.max_pool_size,
            strides=self.max_pool_strides)(block1_output)

        block2 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block1_output)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
        block2 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block2)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)

        block2_output = add([block2, block1_output])
        block2_output = MaxPooling1D(
            pool_size=self.max_pool_size,
            strides=self.max_pool_strides)(block2_output)

        block3 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block2_output)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
        block3 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block3)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)

        block3_output = add([block3, block2_output])
        block3_output = MaxPooling1D(
            pool_size=self.max_pool_size,
            strides=self.max_pool_strides)(block3_output)

        block4 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block3_output)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)
        block4 = SeparableConv1D(
            self.filter_nr,
            kernel_size=self.filter_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_uniform(),
            kernel_regularizer=conv_kern_reg,
            bias_regularizer=conv_bias_reg)(block4)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)

        block4_output = add([block4, block3_output])
        output = GlobalMaxPooling1D()(block4_output)
        output = Dense(self.dense_nr, activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)
        output = Dropout(self.dense_dropout)(output)
        output = Dense(num_classes, activation='softmax')(output)
        model = Model(comment, output)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.Adam(lr=0.01),
            metrics=['accuracy'])
        return model

    def __model_fit(self, model, sentences, labels, path):
        model.fit(
            sentences,
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1)
        model.save(path + "model.hdf5")

    def model_load(self, path):
        return load_model(path + "model.hdf5"), np.load(
            path + "w2i.npy").item(), np.load(path + "l2i.npy").item()

    def fit(self, sentences, labels, path):
        if not os.path.exists(path):
            os.makedirs(path)
        assert len(sentences) == len(
            labels
        )
        print(type(sentences))
        print (type(sentences[0]))
        self.tk.fit_on_texts(sentences)
        labels = self.le.fit_transform(labels)
        sentences = self.tk.texts_to_sequences(sentences)
        sentences = pad_sequences(sentences, maxlen=self.maxlen)
        max_features = max([v for k, v in self.tk.word_index.items()]) + 1
        model = self.__get_model(max_features, self.maxlen,
                                 len(self.le.classes_))
        file = self.__model_fit(model, sentences, labels, path)
        np.save(path + "w2i.npy", self.tk.word_index)
        l2i = {v: k for v, k in enumerate(self.le.classes_)}
        np.save(path + "l2i.npy", l2i)
        return file

    def predict(self, text, model, w2i,
                l2i: dict):
        intent = {}
        tokens = np.asarray([[w2i.get(token, 0) for token in text.split()]])
        padded_tokens = pad_sequences(tokens, maxlen=self.maxlen)
        output = model.predict(padded_tokens)
        intent_ranking = []
        return output[0]


def test_fit():
    import pandas as pd
    data = pd.read_csv("preprocessed_tl.csv")
    sentences = data['description'].values
    labels = data['total_loss'].values
    dpcnn_model = DPCNN()
    dpcnn_model.fit(sentences, labels, "model_2/")
    print("completed test_fit")

def test_predict(samp_doc):
    import json
    def cleanup(doc):
        stoplist = stopwords.words('english')+list(string.punctuation)
        stoplist.remove('no')
        stoplist.remove('not')
        stopset = set(stoplist)
        tokens =  nltk.word_tokenize(doc)
        cleanup = " ".join(filter(lambda word: word not in stopset, doc.split()))
        return cleanup.lower()
    dpcnn_model = DPCNN()
    model, w2i, l2i = dpcnn_model.model_load("model_2/")
    samp_note = cleanup(samp_doc)
    '''
    pred_out = json.dumps(dpcnn_model.predict(samp_note, model, w2i, l2i))
    pred_out = json.loads(pred_out)
    print (samp_doc)
    '''
    print (dpcnn_model.predict(samp_note, model, w2i, l2i))
    #print (pred_out['intent']['name'])
def infer_col(data):
    import json
    from tqdm import tqdm
    dpcnn_model = DPCNN()
    model, w2i, l2i = dpcnn_model.model_load("model_2/")
    res_data = []
    for val in tqdm(data):
        res = np.argmax(dpcnn_model.predict(val, model, w2i, l2i))
        res_data.append(res)
    return res_data



if __name__ == "__main__":
    test_fit()
    samp_str = "ov re iv iv was then pushed over left into ov ds qtr panelrear end smashed incrushednan"
    test_predict(samp_str)