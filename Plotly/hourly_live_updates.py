# -*- coding: utf-8 -*-
"""
testing Dash ability to refresh for live data
"""
import os
import pandas as pd
import requests


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

    return hours

