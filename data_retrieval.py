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
location = 'London%2C%20ENG%2C%20GB'
end_date = '2021-5-14'
iterations = 4
day_chunk = 200

#import existing data to append to
try:
    weather_data = pd.read_csv('Data/weather_data.csv')
except:
    weather_data = pd.DataFrame()

#get data (conform to api extraction limitations)
for i in range(iterations):
    end_date_chunk = dt.datetime.strptime(end_date,'%Y-%m-%d') - dt.timedelta(days= (i * day_chunk) + 1)
    end_date_chunk = dt.datetime.strftime(end_date_chunk,'%Y-%m-%d') 
    
    
    start_date = dt.datetime.strptime(end_date,'%Y-%m-%d') - dt.timedelta(days=((i + 1) * day_chunk))
    start_date = dt.datetime.strftime(start_date,'%Y-%m-%d') 
    
    #url
    url = \
    ''.join(['https://weather.visualcrossing.com/VisualCrossingWebServices/',
             'rest/services/timeline/[LOCATION]/[START_DT]/[END_DT]?',
             'unitGroup=uk&key=[API_KEY]&include=obs'])\
             .replace('[LOCATION]', location)\
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