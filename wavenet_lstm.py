import numpy as np
import pandas as pd
import os
import psutil
import math

#import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15


import time
import datetime

import gc
import seaborn as sns
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

import scipy.signal as sg
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

etq_meta = [
{"start":0,         "end":5656574},
{"start":5656574,   "end":50085878},
{"start":50085878,  "end":104677356},
{"start":104677356, "end":138772453},
{"start":138772453, "end":187641820},
{"start":187641820, "end":218652630},
{"start":218652630, "end":245829585},
{"start":245829585, "end":307838917},
{"start":307838917, "end":338276287},
{"start":338276287, "end":375377848},
{"start":375377848, "end":419368880},
{"start":419368880, "end":461811623},
{"start":461811623, "end":495800225},
{"start":495800225, "end":528777115},
{"start":528777115, "end":585568144},
{"start":585568144, "end":621985673},
{"start":621985673, "end":629145480},
]

df = []
for i in [2, 7, 0, 4, 11, 13, 9, 1, 14, 10]:
    df.append(train[etq_meta[i]['start']:etq_meta[i]['start']+150000*((etq_meta[i]['end'] - etq_meta[i]['start'])//150000)])

train = pd.concat(df)
del df
gc.collect()

num_seg = len(train)//150000
train_X = []
train_y = []
for i in tqdm_notebook(range(num_seg)):
#     train_X.append(fft_process(train['acoustic_data'].iloc[150000 * i:150000 * i + 150000]))
    if 100000 * i + 150000 < len(train):
        train_X.append(train['acoustic_data'].iloc[150000 * i:150000 * i + 150000])
        train_y.append(train['time_to_failure'].iloc[150000 * i + 149999])
del train
gc.collect()
train_X = np.array(train_X,dtype = np.float32)
train_y = np.array(train_y,dtype = np.float32)

X_mean = train_X.mean(0)
X_std = train_X.std(0)
train_X -= X_mean
train_X /= X_std
y_mean = train_y.mean()
y_std = train_y.std()
train_y -= y_mean
train_y /= y_std

test_folder = '../input/test/'
submission = pd.read_csv('../input/sample_submission.csv', index_col = 'seg_id')

test_X = []
test_file_name = []

for seg_name in os.listdir(test_folder):
    #print('seg_name is ' + seg_name)
    seg = pd.read_csv(test_folder + seg_name)
    test_X.append(seg['acoustic_data'])
    test_file_name.append(seg_name[0:-4])
test_X = np.array(test_X,dtype = np.float32)
test_X -= X_mean
test_X /= X_std

train_X = np.expand_dims(train_X,-1)
test_X = np.expand_dims(test_X,-1)

import keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam,SGD
from keras.backend import clear_session
from tensorflow.python.keras import backend as K
import tensorflow as tf

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def keras_model():
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = x
        #         x_ = []
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='tanh',
                              dilation_rate=dilation_rate)(x)
            sigm_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='sigmoid',
                              dilation_rate=dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters=filters,
                       kernel_size=1,
                       padding='same')(x)
            #             x = BatchNormalization()(x)
            #             x = SpatialDropout1D(0.2)(x)
            res_x = Add()([res_x, x])
        #             x_.append(x)
        return res_x

    clear_session()
    inp = Input(shape=(150000, 1))
    Collect = []
    x = wave_block(inp, 16, 3, 8)
    x = AveragePooling1D(10)(x)
    #     x = BatchNormalization()(x)
    #     x = SpatialDropout1D(0.05)(x)
    x = wave_block(x, 32, 3, 5)
    x = AveragePooling1D(10)(x)
    #     x = BatchNormalization()(x)
    #     x = SpatialDropout1D(0.05)(x)
    x = wave_block(x, 64, 3, 3)
    x = AveragePooling1D(10)(x)
    #     x = BatchNormalization()(x)
    #     x = SpatialDropout1D(0.05)(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(150)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1)(x)
    model = Model(inputs=inp, outputs=x)
    return model


model = keras_model()
model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping
def step_decay(epoch):
    x = 0.0001
    if epoch >= 40: x = 0.00001
    return x
lr_decay = LearningRateScheduler(step_decay)

prediction = np.zeros(len(test_X))

model = keras_model()

model.compile(Adam(lr=0.00001), loss='mse', metrics=['mae'])
model.fit(train_X,train_y,
         batch_size = 16,
         epochs=100,
         verbose=2
         )
prediction = model.predict(test_X)[:,0]
submission['seg_id'] = test_file_name
submission['time_to_failure'] = prediction * y_std + y_mean
print(submission.head())
submission.to_csv('submission.csv')
print('Finish')
