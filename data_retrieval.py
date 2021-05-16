# -*- coding: utf-8 -*-
"""
This script use the visualcrossing api to get historical weather data
https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
"""
# =============================================================================
# IMPORTS
# =============================================================================
import requests
import pandas as pd
import datetime as dt

# =============================================================================
# FUNCTIONS
# =============================================================================


# =============================================================================
# PROCESS
# =============================================================================
#load API key
with open('api_key.txt') as doc:
    api_key = doc.read()

#Set other params
LOCATION = 'London%2C%20ENG%2C%20GB'
END_DATE = '2019-3-05'
ITERATIONS = 4
DAY_CHUNK = 200

#import existing data to append to
try:
    weather_data = pd.read_csv('Data/weather_data.csv')
except:
    weather_data = pd.DataFrame()

#get data (conform to api extraction limitations)
for i in range(ITERATIONS):
    end_date_chunk = dt.datetime.strptime(END_DATE,'%Y-%m-%d') - dt.timedelta(days= (i * DAY_CHUNK) + 1)
    end_date_chunk = dt.datetime.strftime(end_date_chunk,'%Y-%m-%d') 
    
    
    start_date = dt.datetime.strptime(END_DATE,'%Y-%m-%d') - dt.timedelta(days=((i + 1) * DAY_CHUNK))
    start_date = dt.datetime.strftime(start_date,'%Y-%m-%d') 
    
    #url
    url = \
    ''.join(['https://weather.visualcrossing.com/VisualCrossingWebServices/',
             'rest/services/timeline/[LOCATION]/[START_DT]/[END_DT]?',
             'unitGroup=uk&key=[API_KEY]&include=obs'])\
             .replace('[LOCATION]', LOCATION)\
             .replace('[START_DT]', start_date)\
             .replace('[END_DT]', end_date_chunk)\
             .replace('[API_KEY]', api_key)

    print('\nExtracting from:\n{}'.format(url))
    
    #make request
    req = requests.get(url)
    req_json = req.json()
    
    #convert to pandas dataframe
    wdays = pd.DataFrame(req_json['days'])
    
    #append to main table
    weather_data = weather_data.append(wdays)
    
#export data to csv
weather_data = weather_data.sort_values(by='datetime')
weather_data.to_csv('Data/weather_data.csv', index=False)