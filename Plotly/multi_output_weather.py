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
                        name = name + '_hist',
                        mode = 'lines+markers+text',
                        marker = dict(color = 'rgba(0,128,0, 1)', size=10, symbol=1,
                                      line = {'width':1}),
                        line = dict(width=3),
                        text=dub[0].values,
                        textposition="top center",
                        texttemplate='%{text:.0f}',
                        textfont_size=12)
                        
    trace2 = go.Scatter(x = dub[1].index,
                        y = dub[1].values,
                        name = name + '_fc',
                        mode = 'lines+markers+text',
                        marker = dict(color = 'rgba(0,0,255, 0.8)', size=15, symbol=5,
                                      line = {'width':1}),
                        line = dict(width=2, dash='longdash'),
                        text=dub[1].values,
                        textposition="top center",
                        texttemplate='%{text:.0f}',
                        textfont_size=12)
    
    return [trace1, trace2]

def plotlyLayout(title, y_label):
    layout = go.Layout(
    title = {'text':title,
             'x':0.5},
    xaxis= {'title':'Date',
            'showgrid':True,
           'gridwidth':1,
           'gridcolor':'rgba(0,0,0,0.05)'},
    yaxis={'title':y_label,
           'showgrid':True,
           'gridwidth':1.5,
           'gridcolor':'rgba(0,0,0,0.15)'},
    legend={'x':0.025, 'y':0.95,
            'bgcolor':'rgba(0,0,0,0)',
            'borderwidth':1.5},
     plot_bgcolor='rgba(227,248,251,0)'
     )
    
    return layout

#DASH
app = dash.Dash()

colors = {'background': '#fff', 'text': '#1E1E1E'}

window_style = {'backgroundColor': colors['background'],
               'width':'49%',
               'display':'inline-block',
               'border-color':'#1e1e1e',
               'border-width':'1px',
               'border-style':'solid'}

flex_grid_col = {
  'display':'flex',
  'justify-content':'space-between',
  'margin':'40px'}

app.layout =\
html.Div(children=[
    html.Div(children=[
        
        html.H1(children='Hello Dash',
                style={'textAlign': 'center','color': colors['text']}),
        html.P(children='Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec risus ligula, consectetur nec metus at, pellentesque dapibus turpis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. In libero nibh, volutpat ut mattis semper, suscipit sodales arcu. Vestibulum tempus porta ex ac sodales. Sed maximus velit risus, quis scelerisque lorem porta eget. Curabitur eu nulla sem. Duis et ullamcorper risus.',
               style={'textAlign': 'left','color': colors['text']})
        ], style={'margin':'40px 40px 20px 40px',
                  'padding':'20px 60px',
                  'border-color':'#1e1e1e',
                  'border-width':'1px',
                  'border-style':'solid'
        }),
    html.Div(children=[

        html.Div(dcc.Graph(id='graph1'), 
            style=window_style),

        html.Div(dcc.Graph(id='graph2'), 
            style=window_style)],
        
    style=flex_grid_col
    ),
        
    html.Div(children=[
        
        html.Div(dcc.Graph(id='graph3'), 
            style=window_style),

        html.Div(dcc.Graph(id='graph4'), 
            style=window_style)],
        
    style=flex_grid_col
    ),
        
        
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
        data = plotlyData(dub=weather[0:2], name='temp'),
        layout = plotlyLayout('Temp C', 'Temp C'))
    
    fig2 = go.Figure(
        data = plotlyData(dub=weather[2:4], name='precip'),
        layout = plotlyLayout('Precip', 'Precip'))
        
    fig3 = go.Figure(
        data = plotlyData(dub=weather[4:6], name='humidity'),
        layout = plotlyLayout('Humidity', 'Humidity')) 
            
    fig4 = go.Figure(
        data = plotlyData(dub=weather[6:], name='windspeed'),
        layout = plotlyLayout('windspeed', 'windspeed')) 
    
    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True, port=8056)