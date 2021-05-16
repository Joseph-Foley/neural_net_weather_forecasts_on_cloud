# -*- coding: utf-8 -*-
"""
script used for building forecast model
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
train = temp.iloc[:-7]
test = temp.iloc[-7:]

#scale data

#make model