# -*- coding: utf-8 -*-
"""
script used for building forecast model
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler#, Normalizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator



# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# EXECUTE
# =============================================================================
#import data
df = pd.read_csv('Data/weather_data.csv')

#get temp and time
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')
df = df.set_index('datetime')

temp = df['temp']

#split data
train = temp.iloc[:-28]
validation = temp.iloc[-28:-7]
test = temp.iloc[-7:]

#scale data
scaler = MinMaxScaler()
scaler.fit(train.values.reshape(-1,1))
train_scaled = scaler.transform(train.values.reshape(-1,1))
validation_scaled = scaler.transform(validation.values.reshape(-1,1))
test_scaled = scaler.transform(test.values.reshape(-1,1))

#make model
length = 30
n_features = 1

model = Sequential()
model.add(GRU(units=16, activation='tanh', input_shape=(length, n_features)))#, stateful=True, batch_input_shape=(1, 30, 1)))
model.add(Dense(1))

#print(model.summary())

#data generator 
generator = TimeseriesGenerator(data=train_scaled, targets=train_scaled, length=length, batch_size=1)
v_generator = None

#callbacks
early_stop = EarlyStopping(monitor='loss', patience=5)

#compile and fit data
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(generator, epochs=8, callbacks=[early_stop])

#evaluate
model.evaluate(generator)

#predict
#model.predict(train_scaled[:7].reshape(-1,7,1))
#model.predict(train_scaled[:11].reshape(-1,11,1))