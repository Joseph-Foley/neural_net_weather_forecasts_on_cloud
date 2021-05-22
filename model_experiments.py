# -*- coding: utf-8 -*-
"""
taking the best model from the GS and then seeing if we can boost performance
with a few tweaks and data transformations
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler#, Normalizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, GRU, Dense, Dropout, LayerNormalization, Bidirectional #BatchNormalization - NO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from model_build_exp import BuildModel

model_name = 'temp_model.h5'

#input hyper parameters of best model
model = BuildModel(model_name, length=90, layers_num=1, layers_type='GRU',
                   units=160, dropout=0.0, batch_size=10, patience=20, epochs=250)

#get data
df = pd.read_csv('Data/weather_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')
df = df.set_index('datetime')
temp = df['temp'].iloc[:-7]
test_data = df['temp'].iloc[-7:]

#setup data
model.setupData(temp)

#load and/or fit the model
model.fitModel()
#model.loadModel()

#get preds
jank = tf.keras.models.load_model(model.model_name)
preds = jank.predict(model.val_generator)#smooth values

#inverse transform
preds = pd.Series(model.scaler.inverse_transform(preds)[:,0],
                  index = model.validation[model.length:].index)

#mae in temp
score = (preds - model.validation[model.length:]).abs().mean()#real values


#score
print('score')
print(score)
