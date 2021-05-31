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

import plotly.express as px
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
    mode = 'lines+markers+text',
    name = 'temp_hist',
    marker = dict(color = 'rgba(0,128,0, 1)', size=10, symbol=1,
                  line = {'width':1}),
    line = dict(width=3),
    text=temp_hist,
    textposition="top center",
    texttemplate='%{text:.0f}',
    textfont_size=14)
    
trace1 = go.Scatter(
    x = temp_fc.index,
    y = temp_fc.values,
    mode = 'lines+markers+text',
    name = 'temp_fc',
    marker = dict(color = 'rgba(0,0,255, 0.8)', size=15, symbol='5',
                  line = {'width':1}),
    line = dict(width=2, dash='longdash'),
    text=temp_fc,
    textposition="top center",
    texttemplate='%{text:.0f}',
    textfont_size=14)

data = [trace0, trace1]  # assign traces to data
layout = go.Layout(
    title = {'text':'Temperature (C) History & Forecast',
             'x':0.5},
    xaxis= {'title':'Date',
            'showgrid':True,
           'gridwidth':1,
           'gridcolor':'rgba(0,0,0,0.05)'},
    yaxis={'title':'Temp C',
           'showgrid':True,
           'gridwidth':1.5,
           'gridcolor':'rgba(0,0,0,0.15)'},
    legend={'x':0.025, 'y':0.95,
            'bgcolor':'rgba(0,0,0,0)',
            'borderwidth':1.5},
     plot_bgcolor='rgba(227,248,251,0)'
)
fig = go.Figure(data=data,layout=layout)
pyo.plot(fig)#, filename='line1.html')


# =============================================================================
# 
# solid = pd.concat([temp_hist, temp_fc], axis=1)
# solid.columns=['hist','fc']
# 
# fig = px.line(solid, spike=2)
# pyo.plot(fig)
# =============================================================================
