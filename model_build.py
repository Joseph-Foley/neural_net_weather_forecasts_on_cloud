# -*- coding: utf-8 -*-
"""
script used for building forecast model
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

temp = df['temp'].iloc[:-7]
test_data = df['temp'].iloc[-7:]
# =============================================================================
# 
# #split data (one year for validation, one week for test)
# train = temp.iloc[:-450]
# validation = temp.iloc[-450:-7]
# test = temp.iloc[-7:]
# 
# #scale data
# scaler = MinMaxScaler()
# scaler.fit(train.values.reshape(-1,1))
# train_scaled = scaler.transform(train.values.reshape(-1,1))
# validation_scaled = scaler.transform(validation.values.reshape(-1,1))
# test_scaled = scaler.transform(test.values.reshape(-1,1))
# 
# #make model
# length = 30
# n_features = 1
# 
# model = Sequential()
# model.add(LSTM(units=100, activation='tanh', input_shape=(length, n_features), dropout=0, recurrent_dropout=0))#, stateful=True, batch_input_shape=(1, 30, 1)))
# #model.add(Bidirectional(LSTM(units=100, activation='tanh', input_shape=(length, n_features), dropout=0, recurrent_dropout=0)))
# #model.add(LayerNormalization())
# model.add(Dense(1))
# 
# #print(model.summary())
# 
# #data generator 
# generator = TimeseriesGenerator(data=train_scaled, targets=train_scaled, length=length, batch_size=1)
# val_generator = TimeseriesGenerator(data=validation_scaled, targets=validation_scaled, length=length, batch_size=1)
# 
# #callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=5)
# 
# #compile and fit data
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(generator, validation_data=val_generator, epochs=8, callbacks=[early_stop])
# 
# #evaluate
# model.evaluate(val_generator)
# =============================================================================

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
    units - number of units in rnn layers
    num_step_preds - number of steps/days in time to predict
    dropout - dropout % to be applied to rnn units
    batch_size - number of samples to feed model at a time.
    patience - how many epochs to wait before stopping model after finding good score.
    """
    def __init__(self, length=10, layers_num=1, layers_type='LSTM',\
                 units=50, num_step_preds=1, dropout=0.0, epochs=8,\
                 batch_size=1, patience=5):
        
        #assertions for input
        assert 0 < layers_num < 4, "1 <= layers_num <= 3"
        assert layers_type in ['LSTM', 'GRU'], "layers_type is LSTM or GRU"
        assert 0 <= dropout < 1, "dropout must be float < 1"
        
        #initialise
        self.length = length
        self.layers_num = layers_num
        self.layers_type = layers_type
        self.units = units
        self.num_step_preds = num_step_preds
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = 1
        
        #callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        
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
    
    def plotPreds(self, predictions, test_series=None, run_up=None, ylabel='units'):
        pass
    
test = BuildModel(units=10, epochs=2)
test.setupData(temp)
test.fitModel()   

print(test.model.history.history)
predictions = test.predAhead(7)

#plotting
plt.figure(figsize=(12, 8))
plt.ylabel('temp')
plt.xlabel('datetime')
plt.plot(test_data)
plt.scatter(predictions.index, predictions, edgecolors='k', label='predictions', c='#2ca02c', s=64)
plt.scatter(test_data.index, test_data, marker='X', edgecolors='k', label='test_data',
                  c='#ff7f0e', s=200)


def gridTableGen(length: list, layers_num: list, layers_type: list,\
               units: list,  dropout: list):
    """returns table of every combo for the hyperparameters"""
    
    #get cross joins to acquire every combination
    grid_table = pd.DataFrame(length).merge(\
                 pd.DataFrame(layers_num), how='cross').merge(\
                 pd.DataFrame(layers_type), how='cross').merge(\
                 pd.DataFrame(units), how='cross').merge(\
                 pd.DataFrame(dropout), how='cross')
                                                     
    grid_table.columns = \
        ['length', 'layers_num', 'layers_type', 'units', 'dropout']
        
    return grid_table



def gridSearch(grid_table, data):
    """searches through hyperparameters in grid_table to determine optimium model"""
    #record time for file_name
    time_now = str(round(time.time()))
        
    #make results table to append results onto
    results_cols =\
        pd.DataFrame(columns=['loss', 'mae', 'val_loss', 'val_mae', 'epochs'])
        
    results_table = pd.concat([grid_table, results_cols], axis=1)
    
    #iterate through the table and fit the models
    for i, row in grid_table.iterrows():
        #input hyperparameters
        print('\nNow Training \n{}'.format(row.to_dict()))
        grid_mod = \
            BuildModel(length=row['length'], layers_num=row['layers_num'],\
                       layers_type=row['layers_type'],units=row['units'],\
                       num_step_preds=1, dropout=row['dropout'], epochs=2,\
                       batch_size=10, patience=5)
        
        #setup data and train the model
        grid_mod.setupData(data)
        grid_mod.fitModel()
        
        #find best epoch (val_mae)
        hist = grid_mod.history
        best_epoch = hist[hist['val_mae'] == hist['val_mae'].min()]\
                     .iloc[:1]
        
        #update results table
        results_table.loc[i, ['loss', 'mae', 'val_loss', 'val_mae']] =\
            best_epoch.values[0].round(4)
        
        results_table.loc[i, 'epochs'] = best_epoch.index[0]
        
        #save to drive
        results_table.to_csv('results_table_' + time_now + '.csv', index=False)
        
    return results_table

# =============================================================================
# #grid search
# #grid table and results table
# length = [1,5]
# layers_num = [1,2,3]
# layers_type = ['GRU', 'LSTM']
# units = [10, 20] 
# dropout = [0.0, 0.2]
# 
# grid_table = gridTableGen(length, layers_num, layers_type, units, dropout)
# results = gridSearch(grid_table, temp)
# 
# =============================================================================
