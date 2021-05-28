# -*- coding: utf-8 -*-
"""
Creates a line plot in of the history and the forecast in plotly
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import datetime as dt

import plotly.offline as pyo
import plotly.graph_objs as go


# =============================================================================
# EXECUTE
# =============================================================================
#import data
df = pd.read_csv('../Data/weather_data.csv')

#get temp and time
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')
df = df.set_index('datetime')

temp_hist = df['temp'].iloc[-14:-6]
temp_fc = df['temp'].iloc[-7:]

#graphing
# Create traces
trace0 = go.Scatter(
    x = temp_hist.index,
    y = temp_hist.values,
    mode = 'lines+markers',
    name = 'temp_hist')
    
trace1 = go.Scatter(
    x = temp_fc.index,
    y = temp_fc.values,
    mode = 'lines+markers',
    name = 'temp_fc')

data = [trace0, trace1]  # assign traces to data
layout = go.Layout(
    title = 'Temperature (C) History & Forecast'
)
fig = go.Figure(data=data,layout=layout)
pyo.plot(fig)#, filename='line1.html')
