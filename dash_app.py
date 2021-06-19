# -*- coding: utf-8 -*-
"""
same as the 2x2 but now we use interval.
shows a multi output callback decorator
uses weather api
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

import pandas as pd
import numpy as np
import datetime as dt
import requests
import os

import plotly.offline as pyo
import plotly.graph_objs as go

from model_build_smooth import BuildModel


def getData():
    """API import. Return last 30 days of weather in a DataFrame"""
    #load entrant (for Heroku)
    entrant = 'VFJ5W4L3FNJLNDEMWN6JZSEWB'
    
    #form url
    url = ''.join(['https://weather.visualcrossing.com/',
                   'VisualCrossingWebServices/rest/services/timeline/',
                   'London%2C%20ENG%2C%20GB/last30days?unitGroup=uk&key={}',
                   '&include=obs'])\
                   .format(entrant)
                   
    #make request
    req = requests.get(url)
    req_json = req.json()
    
    #convert to pandas dataframe
    wdays = pd.DataFrame(req_json['days'])
    
    #ensure its 30 days long
    df = wdays.iloc[(len(wdays)-30):]
    
    #get temp and time
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df = df.set_index('datetime')
    
    return df

def loadModels():
    """Instantiates model class and then loads h5 model"""
    #file names
    files = os.listdir('./Colab_Models')
    files = ['./Colab_Models/' + file for file in files]
    
    #put temperature first
    files[0] , files[2] = files[2] , files[0]
    
    #assert we have 4 h5 models
    assert len(files) == 4
    assert files[0][-3:] == '.h5'
    
    #instantiate model classes
    model_dict = \
        {key : BuildModel(model_name=key, length=30) for key in files}
        
    #load h5 models
    for key in model_dict.keys():
        model_dict[key].loadModel()
        
    return model_dict

def plotlyData(name: str, hist, fc):
    """plots history and forecast"""
    
    trace1 = go.Scatter(x=hist.index,
                        y=hist.values,
                        name='History',
                        mode='lines+markers+text',
                        marker=dict(color='rgba(0,128,0, 1)', size=10, symbol=1,
                                    line={'width':1}),
                        line=dict(width=3),
                        text=hist.values,
                        textposition="top center",
                        texttemplate='%{text:.0f}',
                        textfont_size=12)
    
    trace2 = go.Scatter(x=fc.index,
                        y=fc.values,
                        name='Forecast',
                        mode='lines+markers+text',
                        marker=dict(color = 'rgba(0,0,255, 0.8)', size=15, symbol=5,
                                    line = {'width':1}),
                        line=dict(width=2, dash='longdash'),
                        text=fc.values,
                        textposition="top center",
                        texttemplate='%{text:.0f}',
                        textfont_size=12)
    
    return [trace1, trace2]

def plotlyLayout(title, y_label):
    
    layout = go.Layout(
        
    title={'text':title,
             'x':0.5},
    xaxis={'title':'Date',
            'showgrid':True,
           'gridwidth':1,
           'gridcolor':'rgba(0,0,0,0.05)'},
    yaxis={'title':y_label,
           'showgrid':True,
           'gridwidth':1.5,
           'gridcolor':'rgba(0,0,0,0.15)'},
    legend={'x':0.025, 'y':0.95,
            'bgcolor':'rgba(255,255,255,1)',
            'borderwidth':0.5},
     plot_bgcolor='rgba(227,248,251,0)'
     
                      )
    return layout

#DASH
app = dash.Dash()
server = app.server

colors = {'background': '#111111', 'text': '#7FDBFF'}

window_style = {'backgroundColor': colors['background'],
               'width':'50%','display':'inline-block'}

model_dict = loadModels()

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
       interval=6*60*60*1000, # 6hr * 60mins * 60secs * 1000milisecs
       n_intervals=0 )   
                    ],
        
style={'backgroundColor': colors['background']})

@app.callback([Output('graph1','figure'),
               Output('graph2','figure'),
               Output('graph3','figure'),
               Output('graph4','figure')],
          [Input('interval-component', 'n_intervals')])
def updateGraphs(n):
    #refresh data from the api
    df = getData()
    
    #make predictions
    model_dict = loadModels()
    dict_keys = list(model_dict.keys())
    metrics = ['temp', 'precip', 'humidity', 'windspeed']
    
    preds = \
    [model_dict[dict_keys[i]].predAhead(7, df[metrics[i]]) for i in range(4)]
    
    #format predictions
    preds = pd.DataFrame(preds, index=metrics).T\
            .round(2)
            
    preds[preds < 0] = 0
    
    #append first pred to hist for continuous line chart.
    df = df.append(preds.iloc[:1])
    
    
    fig0 = go.Figure(
        data=plotlyData(name=metrics[0],\
                        hist=df[metrics[0]].iloc[-8:], fc=preds[metrics[0]]),
        layout=plotlyLayout('Temperature \N{DEGREE SIGN}C',\
                            'Temperature \N{DEGREE SIGN}C'))
    
    fig1 = go.Figure(
        data = plotlyData(name=metrics[1],\
                          hist=df[metrics[1]].iloc[-8:], fc=preds[metrics[1]]),
        layout=plotlyLayout('Precipitation (mm)',\
                            'Precipitation (mm)'))
        
    fig2 = go.Figure(
        data = plotlyData(name=metrics[2],\
                          hist=df[metrics[2]].iloc[-8:], fc=preds[metrics[2]]),
        layout=plotlyLayout('Humidity (%)',\
                            'Humidity (%)'))
            
    fig3 = go.Figure(
        data = plotlyData(name=metrics[3],\
                          hist=df[metrics[3]].iloc[-8:], fc=preds[metrics[3]]),
        layout=plotlyLayout('Windspeed (mph)',\
                            'Windspeed (mph)'))
    
    return fig0, fig1, fig2, fig3

if __name__ == '__main__':
    #model_dict = loadModels()
    app.run_server()