# -*- coding: utf-8 -*-
"""
Script to play around with Dash
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import datetime as dt

import plotly.offline as pyo
import plotly.graph_objs as go

#PLOTLY
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


#DASH
app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(
        children='Dash: A web application framework for Python.',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': data,
            'layout': layout
        }
    )],
    style={'backgroundColor': colors['background']}
)



#RUN
app.run_server()
