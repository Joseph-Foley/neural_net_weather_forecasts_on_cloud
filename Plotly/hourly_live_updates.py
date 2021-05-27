# -*- coding: utf-8 -*-
"""
testing Dash ability to refresh for live data
"""
import os
import pandas as pd
import requests

import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def getSeries():
    #load API key
    with open('../api_key.txt') as doc:
        api_key = doc.read()
        
    URL =\
    'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/London%2C%20ENG%2C%20GB/today?unitGroup=uk&key={}&include=obs%2Chours'
    
    URL = URL.format(api_key)
    
    #make request
    req = requests.get(URL)
    req_json = req.json()
    
    #convert to pandas dataframe
    hours = pd.DataFrame(req_json['days'][0]['hours'])
    
    #drop where no temp
    hours = hours[hours['temp'].notna()]
    
    return hours

def plotlyData(hours):
    trace = go.Scatter( x = hours['datetime'],
                        y = hours['temp'],
                        mode = 'lines+markers',
                        name = 'temp_hist')
    
    return trace

#layout = go.Layout(title = 'Temperature hourly (live)')


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

    dcc.Graph(id='live-update-graph',
              style={'backgroundColor': colors['background']}
            ),
    dcc.Interval(
        id='interval-component',
        interval=1*60*60*1000, # 1hr * 60mins * 60secs * 1000milisecs
        n_intervals=0
    )])

@app.callback(Output('live-update-graph','figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    hours = getSeries()
    
    fig = go.Figure(
        data = [plotlyData(hours)])
    return fig


if __name__ == '__main__':
    app.run_server()
