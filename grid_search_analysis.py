# -*- coding: utf-8 -*-
"""
analyse the results of the gridsearch

'length', 'layers_num', 'layers_type', 'units', 'dropout'
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

gs = pd.read_csv('Data/results_table_temp.csv')

gs.groupby('length')['val_mae'].mean().plot(kind='bar', title='mean'); plt.show()
gs.groupby('layers_num')['val_mae'].mean().plot(kind='bar', title='mean'); plt.show()
gs.groupby('layers_type')['val_mae'].mean().plot(kind='bar', title='mean'); plt.show()
gs.groupby('units')['val_mae'].mean().plot(kind='bar', title='mean'); plt.show()
gs.groupby('dropout')['val_mae'].mean().plot(kind='bar', title='mean'); plt.show()

gs.groupby('length')['val_mae'].mean().plot(kind='bar', title='min'); plt.show()
gs.groupby('layers_num')['val_mae'].mean().plot(kind='bar', title='min'); plt.show()
gs.groupby('layers_type')['val_mae'].mean().plot(kind='bar', title='min'); plt.show()
gs.groupby('units')['val_mae'].mean().plot(kind='bar', title='min'); plt.show()
gs.groupby('dropout')['val_mae'].mean().plot(kind='bar', title='min'); plt.show()
"""
Whilst the MAE isnt much better than the baseling, the model is picking up a trend.
We see this by its increasing performance as we feed it longer sequences. 
The model is learning to predict the future better by being fed more history at a time.
Dropout does not improve the model performance. Its not often dropout is used in RNN networks, they tend to overfit for other reasons.
LSTM is just slightly better than GRU.
Increasing the number of RNN layers does not improve the predictive power of the model.
"""