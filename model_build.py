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
from tensorflow.keras.layers import InputLayer, LSTM, GRU, Dense, Dropout, LayerNormalization, Bidirectional #BatchNormalization - NO
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

#split data (one year for validation, one week for test)
train = temp.iloc[:-450]
validation = temp.iloc[-450:-7]
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
model.add(LSTM(units=100, activation='tanh', input_shape=(length, n_features), dropout=0, recurrent_dropout=0))#, stateful=True, batch_input_shape=(1, 30, 1)))
#model.add(Bidirectional(LSTM(units=100, activation='tanh', input_shape=(length, n_features), dropout=0, recurrent_dropout=0)))
#model.add(LayerNormalization())
model.add(Dense(1))

#print(model.summary())

#data generator 
generator = TimeseriesGenerator(data=train_scaled, targets=train_scaled, length=length, batch_size=1)
val_generator = TimeseriesGenerator(data=validation_scaled, targets=validation_scaled, length=length, batch_size=1)

#callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5)

#compile and fit data
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(generator, validation_data=val_generator, epochs=8, callbacks=[early_stop])

#evaluate
model.evaluate(val_generator)

#predict
#model.predict(train_scaled[:7].reshape(-1,7,1))
#model.predict(train_scaled[:11].reshape(-1,11,1))

#model class for training
#TODO assert length is not > than validation gen
class BuildModel():
    """
    Build a model. Arguments allow one to customise the hyper parameters
    ATTRIBUTES :- 
    length - number of steps in time sequence to feed the rnn
    layers_num - number of rnn layers in model (capped at 3)
    layers_type - select "LSTM" or "GRU"
    units_rnn - number of units in rnn layers
    units_dense - number of units in final dense (used for multi step preds)
    dropout - dropout % to be applied to rnn units
    batch_size - number of samples to feed model at a time.
    patience - how many epochs to wait before stopping model after finding good score.
    """
    def __init__(self, length=10, layers_num=1, layers_type='LSTM',\
                 units_rnn=50, units_dense=1, dropout=0.0, epochs=8,\
                 batch_size=1, patience=5):
        
        #assertions for input
        assert 0 < layers_num < 4, "1 <= layers_num <= 3"
        assert layers_type in ['LSTM', 'GRU'], "layers_type is LSTM or GRU"
        assert 0 <= dropout < 1, "dropout must be float < 1"
        
        #initialise
        self.length = length
        self.layers_num = layers_num
        self.layers_type = layers_type
        self.units_rnn = units_rnn
        self.units_dense = units_dense
        self.dropout = dropout
        self.epochs = epochs
        n_features = 1
        
        #callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
        
        #BUILD MODEL
        ##inputs
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.length, n_features)))
        
        ##add extra layers as required (or not if layers_num = 1)
        for i in range(layers_num - 1):
            self.model.add(eval('{}(units={}, dropout={}, return_sequences=True)'\
                .format(self.layers_type, self.units_rnn, self.dropout)))
                
        ##closing rnn layer (do not return squences)
        self.model.add(eval('{}(units={}, dropout={})'\
                .format(self.layers_type, self.units_rnn, self.dropout)))
            
        ##Dense output
        self.model.add(Dense(units=self.units_dense))
                       
        #compile model
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        #provide summary
        #TODO remove ()
        self.summary = self.model.summary()
        
    def fitModel(self, generator, val_generator):
        """
        Fits the model on your generators for training and validation sets.
        EarlyStopping call back ends training if val_loss doesnt improve
        """
        self.model.fit(generator, validation_data=val_generator,\
                       epochs=self.epochs, callbacks=self.callbacks)
            
   