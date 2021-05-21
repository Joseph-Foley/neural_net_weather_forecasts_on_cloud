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

#load or fit the model
#model.fitModel()
model.loadModel('temp_model.h5')