# -*- coding: utf-8 -*-
"""
add experimental features to the model to see if it boost performance
"""
# =============================================================================
# IMPORTS
# =============================================================================
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

class BuildModel():
    """
    Build a model. Arguments allow one to customise the hyper parameters
    ATTRIBUTES :- 
    length - number of steps in time sequence to feed the rnn
    layers_num - number of rnn layers in model (capped at 3)
    layers_type - select "LSTM" or "GRU"
    units - number of units in rnn layers
    num_step_preds - number of steps/days in time to predict
    dropout - dropout % to be applied to rnn units
    batch_size - number of samples to feed model at a time.
    patience - how many epochs to wait before stopping model after finding good score.
    model_name - file name of model we save. must end in ".h5" eg 'temp_model.h5'
    """
    def __init__(self, model_name, length=10, layers_num=1, layers_type='LSTM',\
                 units=50, num_step_preds=1, dropout=0.0, epochs=8,\
                 batch_size=1, patience=5):
        
        #assertions for input
        assert 0 < layers_num < 4, "1 <= layers_num <= 3"
        assert layers_type in ['LSTM', 'GRU'], "layers_type is LSTM or GRU"
        assert 0 <= dropout < 1, "dropout must be float < 1"
        assert model_name[-3:] == '.h5', "End model_name with '.h5'"
        
        #initialise
        self.length = length
        self.layers_num = layers_num
        self.layers_type = layers_type
        self.units = units
        self.num_step_preds = num_step_preds
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = model_name
        self.n_features = 1
        
        #callbacks
        self.callbacks =[EarlyStopping(monitor='val_loss', patience=patience),\
                         ModelCheckpoint(self.model_name, monitor='val_loss',\
                                         save_best_only=True)]
        
        #BUILD MODEL
        ##inputs
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.length, self.n_features)))
        
        ##add extra layers as required (or not if layers_num = 1)
        for i in range(layers_num - 1):
            self.model.add(eval('{}(units={}, dropout={}, return_sequences=True)'\
                .format(self.layers_type, self.units, self.dropout)))
                
        ##closing rnn layer (do not return squences)
        self.model.add(eval('{}(units={}, dropout={})'\
                .format(self.layers_type, self.units, self.dropout)))
            
        ##Dense output
        self.model.add(Dense(units=self.num_step_preds))
                       
        #compile model
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def setupData(self, series, val_days=450):
        """
        splits data, scales data, creates generators for the model
        """
        assert val_days > self.length , "val_days must exceed lenght"
        
        #split data into train and validation
        self.train = series.iloc[:-val_days]
        self.validation = series.iloc[-val_days:]
        
        #scale data for neural network suitability
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train.values.reshape(-1,1))
        
        self.train_scaled = \
            self.scaler.transform(self.train.values.reshape(-1,1))
        
        self.validation_scaled = \
             self.scaler.transform(self.validation.values.reshape(-1,1))
        
        #create time series generators
        self.generator = \
             TimeseriesGenerator(data=self.train_scaled,\
                                 targets=self.train_scaled,\
                                 length=self.length,\
                                 batch_size=self.batch_size)
                 
        self.val_generator = \
             TimeseriesGenerator(data=self.validation_scaled,\
                                 targets=self.validation_scaled,\
                                 length=self.length,\
                                 batch_size=self.batch_size)

    def fitModel(self):
        """
        Fits the model on your generators for training and validation sets.
        EarlyStopping call back ends training if val_loss doesnt improve.
        Record epoch metrics in a DataFrame.
        """
        self.model.fit(self.generator, validation_data=self.val_generator,\
                       epochs=self.epochs, callbacks=self.callbacks)
            
        self.history = pd.DataFrame(self.model.history.history)
        
    def loadModel(self):
        """
        Load a model instead of fitting a new one (uses model_name)
        """
        self.model = tf.keras.models.load_model(self.model_name)
            
    def predAhead(self, days, series=None):
        """
        Predicts a number of days ahead set by the user. Input your own
        series or dont if you want to predict off of the validation set.
        """
        assert self.num_step_preds == 1,\
            "sorry, function not yet available for multi step models"
        
        #use end of the validation set to project forward if no series given
        if series == None:
            series = self.validation
        
        #get end of the series to plug into the model
        assert len(series) >= self.length,\
            "series must be at least {} days".format(self.length)
            
        series_cut = series.iloc[-self.length:]
        
        #scale inputs to what model is expecting    
        series_scaled = \
            self.scaler.transform(series_cut.values.reshape(-1,1))
            
        #predict ahead by appending predictions and removing first values
        pred_series = series_scaled.reshape(1, self.length, self.n_features)
        predictions = []
        
        for i in range(days):
            pred = self.model.predict(pred_series)
            pred_series = np.append(pred_series[:,1:,:], [pred], axis=1)
            predictions.append(pred)
            
        #inverse scale back to original units
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(\
                           predictions.reshape(days, self.n_features))\
                          .round(1)
        
        #convert to pandas series
        predictions = pd.Series(predictions.reshape(days))
        predictions.index = self.validation.index[-days:] +\
                                 dt.timedelta(days=days)
            
        return predictions
    
    def plotPreds(self, predictions, test_series=None, run_up=None,\
                  ylabel='units'):
        """
        plot the predictions of the model. plot them against another series
        (test series). plot with with a run up leading to the pred period
        (validation set).
        """
        #set up figure
        plt.figure(figsize=(10,6))
        plt.ylabel(ylabel)
        plt.xlabel('datetime')
        
        #plot lines
        if run_up is None:
            run_up = self.validation[-7:]
            
        if test_series is not None:
            plt.plot(pd.concat([run_up, test_series[:1]]))
            plt.plot(test_series)
            
        else:
            plt.plot(run_up)
            
        #plot points
        plt.scatter(predictions.index, predictions, edgecolors='k',\
                    label='predictions', c='#2ca02c', s=64)
            
        if test_series is not None:
            plt.scatter(test_series.index, test_series, marker='X',\
                        edgecolors='k', label='test_data', c='#ff7f0e', s=200)
                
        plt.legend()

    