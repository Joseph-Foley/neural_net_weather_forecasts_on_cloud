# -*- coding: utf-8 -*-
"""
This script use the visualcrossing api to get historical weather data
https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
"""
# =============================================================================
# IMPORTS
# =============================================================================
import requests

# =============================================================================
# FUNCTIONS
# =============================================================================


# =============================================================================
# PROCESS
# =============================================================================
#load API key
with open('api_key.txt') as f:
    api_key = f.read()
    
#url
url =\
"""
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/
services/timeline/London%2C%20ENG%2C%20GB/2021-5-12/2021-5-13?
unitGroup=uk&key=[API_KEY]&include=obs
""".replace('\n', '')\
   .replace('[API_KEY]', api_key)

#make request
req = requests.get(url)