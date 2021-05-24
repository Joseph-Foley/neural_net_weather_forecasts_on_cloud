# -*- coding: utf-8 -*-
"""
Now adds smoothing to the series. Smoothing parameters are added to our models
hyperparameters.
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

from scipy.ndimage import gaussian_filter1d

# =============================================================================
# CLASSES & FUNCTIONS
# =============================================================================

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
    g_filt - gaussian filter for smoothing. Default: no smoothing
    batch_size - number of samples to feed model at a time.
    patience - how many epochs to wait before stopping model after finding good score.
    model_name - file name of model we save. must end in ".h5" eg 'temp_model.h5'
    """
    def __init__(self, model_name, length=10, layers_num=1, layers_type='LSTM',\
                 units=50, dropout=0.0, g_filt=00.1, num_step_preds=1,\
                 epochs=8, batch_size=1, patience=5):
        
        #assertions for input
        assert 0 < layers_num < 4, "1 <= layers_num <= 3"
        assert layers_type in ['LSTM', 'GRU'], "layers_type is LSTM or GRU"
        assert 0 <= dropout < 1, "dropout must be float < 1"
        assert model_name[-3:] == '.h5', "End model_name with '.h5'"
        
        #initialise
        self.model_name = model_name        
        self.length = length
        self.layers_num = layers_num
        self.layers_type = layers_type
        self.units = units
        self.num_step_preds = num_step_preds
        self.dropout = dropout
        self.g_filt = g_filt
        self.epochs = epochs
        self.batch_size = batch_size
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
        
        #Apply smoothing filters  
        self.train_smooth = \
             gaussian_filter1d(self.train, self.g_filt)\
                 .reshape(-1,1)
            
        self.validation_smooth = \
             gaussian_filter1d(self.validation, self.g_filt)\
                 .reshape(-1,1)

        #create time series generators
        self.generator = \
             TimeseriesGenerator(data=self.train_smooth,\
                                 targets=self.train_smooth,\
                                 length=self.length,\
                                 batch_size=self.batch_size)
                 
        self.val_generator = \
             TimeseriesGenerator(data=self.validation_smooth,\
                                 targets=self.validation_smooth,\
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


def gridTableGen(length: list, layers_num: list, layers_type: list,\
                 units: list, g_filt: list):
    """returns table of every combo for the hyperparameters"""
    
    #get cross joins to acquire every combination
    grid_table = pd.DataFrame(length).merge(\
                 pd.DataFrame(layers_num), how='cross').merge(\
                 pd.DataFrame(layers_type), how='cross').merge(\
                 pd.DataFrame(units), how='cross').merge(\
                 pd.DataFrame(g_filt), how='cross')  
                                                          
    grid_table.columns = \
        ['length', 'layers_num', 'layers_type', 'units', 'g_filt']
        
    return grid_table

def gridSearch(grid_table, data):
    """searches through hyperparameters in grid_table to determine optimium model"""
    #record time for file_name
    time_now = str(round(time.time()))
        
    #make results table to append results onto
    results_cols =\
        pd.DataFrame(columns=['loss', 'mae', 'val_loss', 'val_mae',\
                              'val_mae_og','epochs'])
        
    results_table = pd.concat([grid_table, results_cols], axis=1)
    
    #iterate through the table and fit the models
    for i, row in grid_table.iterrows():
        #input hyperparameters
        print('\nNow Training ({})\n{}'.format(i, row.to_dict()))
        grid_mod = \
            BuildModel(model_name='temp_model.h5', length=row['length'],\
                       layers_num=row['layers_num'], \
                       layers_type=row['layers_type'], units=row['units'],\
                       g_filt=row['g_filt'], num_step_preds=1,\
                       epochs=2, batch_size=10, patience=5)
        
        #setup data and train the model
        grid_mod.setupData(data)
        grid_mod.fitModel()
        
        #find best epoch (val_mae)
        hist = grid_mod.history
        best_epoch = hist[hist['val_mae'] == hist['val_mae'].min()]\
                     .iloc[:1]
                     
        #calculate val_mae in unsmoothed original units
        best_model = tf.keras.models.load_model(grid_mod.model_name)
        preds = best_model.predict(grid_mod.val_generator)
        preds = pd.Series(preds[:,0],\
                    index = grid_mod.validation[grid_mod.length:].index)

        val_mae_og = (preds - grid_mod.validation[grid_mod.length:]).abs()\
                     .mean()
        
        #update results table
        results_table.loc[i, ['loss', 'mae', 'val_loss', 'val_mae']] =\
            best_epoch.values[0].round(4)
        
        results_table.loc[i, 'epochs'] = best_epoch.index[0]
        results_table.loc[i, 'val_mae_og'] = val_mae_og
        
        #save to drive
        results_table.to_csv('results_table_' + time_now + '.csv', index=False)
        
    return results_table

def fastSearch(data: pd.Series, length: list, layers_num: list,\
               layers_type: list, units: list, g_filt: list, model_name: str,\
               best_dict=None):
    """
    First it will set all hyperparameters to their first value in the lists we
    pass in.
    Then list by list it will train the model, keeping the best performing
    element in that list.
    Its recommended that you pass in the resulting dictionary into this 
    function a second time.
    """
    #record time for file_name
    time_now = str(round(time.time()))
    
    #set initial values if no specified parameters given.
    if best_dict is None:
        best_dict = {}
        
        best_dict['length'] = [length[0], length]
        best_dict['layers_num'] = [layers_num[0], layers_num]
        best_dict['layers_type'] = [layers_type[0], layers_type]
        best_dict['units'] = [units[0], units]
        best_dict['g_filt'] = [g_filt[0], g_filt]
    
    records = pd.DataFrame()
    
    #go through each hyperparameter
    for key in best_dict.keys():
        if len(best_dict[key][1]) == 0:
            continue
        
        scores = []
        
        #go through each value
        for item in best_dict[key][1]:
            best_dict[key][0] = item
            
            model = \
                BuildModel(model_name=model_name,\
                           length=best_dict['length'][0],\
                           layers_num=best_dict['layers_num'][0], \
                           layers_type=best_dict['layers_type'][0],\
                           units=best_dict['units'][0],\
                           g_filt=best_dict['g_filt'][0], num_step_preds=1,\
                           epochs=2, batch_size=10, patience=5)
                    
            #setup data and train the model
            model.setupData(data)
            model.fitModel()
                         
            #calculate val_mae in unsmoothed original units
            best_model = tf.keras.models.load_model(model_name)
            preds = best_model.predict(model.val_generator)
            preds = pd.Series(preds[:,0],\
                        index = model.validation[model.length:].index)
        
            val_mae_og = (preds - model.validation[model.length:]).abs()\
                         .mean()
            
            record = pd.DataFrame(best_dict).iloc[:1] 
            record['val_mae_og'] = val_mae_og
            
            #append score
            scores.append(val_mae_og)
            records = records.append(record)
            records.to_csv('records_' + time_now + '.csv', index=False)
            
        #get param value that performed the best
        best_score = min(scores)
        best_dict[key][0] = best_dict[key][1][scores.index(best_score)]
        
    return records, best_dict
    
# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
#grid search
#grid table and results table
    #import data
    df = pd.read_csv('Data/weather_data.csv')
    
    #get temp and time
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')
    df = df.set_index('datetime')
    
    temp = df['temp'].iloc[:-7]
    test_data = df['temp'].iloc[-7:]


    length = [15, 30, 60]
    layers_num = [1, 2]
    layers_type = ['LSTM']
    units = [20, 40, 80] 
    g_filt = [0.5, 0.75, 1.0, 1.25]
    
    model_name = 'temp_model.h5'
    
    #grid_table = gridTableGen(length, layers_num, layers_type, units, g_filt)
    #results = gridSearch(grid_table, temp)
    
    records, best_dict = fastSearch(temp, length, layers_num, layers_type, units, g_filt, model_name='test.h5', best_dict=None)
    records2, best_dict2 = fastSearch(temp, length, layers_num, layers_type, units, g_filt, model_name='test.h5', best_dict=best_dict)
    
    records_all = pd.concat([records,records2])