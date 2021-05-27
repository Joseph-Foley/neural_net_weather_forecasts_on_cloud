# -*- coding: utf-8 -*-
"""
same as the 2x2 but now we use interval.
shows a multi output callback decorator
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

import pandas as pd
import numpy as np
import datetime as dt

import plotly.offline as pyo
import plotly.graph_objs as go

def getSeries():
    """csv import and return the 4 vars hist and fc"""
    #import data
    df = pd.read_csv('../Data/weather_data.csv')
    
    #get temp and time
    #df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')
    df = df.set_index('datetime')
    
    weather = [df['temp'].iloc[-14:-6],
               df['temp'].iloc[-7:],
               df['precip'].iloc[-14:-6],
               df['precip'].iloc[-7:],
               df['humidity'].iloc[-14:-6],
               df['humidity'].iloc[-7:],
               df['windspeed'].iloc[-14:-6],
               df['windspeed'].iloc[-7:]]
    
    return weather

def plotlyData(dub: list, name: str):
    assert len(dub) == 2
    
    trace1 = go.Scatter(x = dub[0].index,
                        y = dub[0].values,
                        mode = 'lines+markers',
                        name = name + '_hist')
    
    trace2 = go.Scatter(x = dub[1].index,
                        y = dub[1].values,
                        mode = 'lines+markers',
                        name = name + '_fc')
    
    return [trace1, trace2]

#DASH
app = dash.Dash()

colors = {'background': '#111111', 'text': '#7FDBFF'}

window_style = {'backgroundColor': colors['background'],
               'width':'50%','display':'inline-block'}

app.layout =\
html.Div(children=[
    html.H1(children='Hello Dash',\
            style={'textAlign': 'center','color': colors['text']}),

    html.Div(dcc.Graph(id='graph1'), 
            style=window_style),

    html.Div(dcc.Graph(id='graph2'), 
            style=window_style),

    html.Div(dcc.Graph(id='graph3'), 
            style=window_style),

    html.Div(dcc.Graph(id='graph4'), 
            style=window_style),

    dcc.Interval(
       id='interval-component',
       interval=1*60*60*1000, # 1hr * 60mins * 60secs * 1000milisecs
       n_intervals=0 )   
                    ],
        
style={'backgroundColor': colors['background']})

@app.callback([Output('graph1','figure'),
               Output('graph2','figure'),
               Output('graph3','figure'),
               Output('graph4','figure')],
          [Input('interval-component', 'n_intervals')])
def updateGraphs(n):
    weather = getSeries()
    
    fig1 = go.Figure(
        data = plotlyData(dub=weather[0:2], name='temp'))
    
    fig2 = go.Figure(
        data = plotlyData(dub=weather[2:4], name='precip'))
        
    fig3 = go.Figure(
        data = plotlyData(dub=weather[4:6], name='humidity'))
            
    fig4 = go.Figure(
        data = plotlyData(dub=weather[6:], name='windspeed'))
    
    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True, port=8056)