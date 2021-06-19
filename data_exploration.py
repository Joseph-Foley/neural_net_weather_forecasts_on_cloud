# -*- coding: utf-8 -*-
"""
Exporatory analysis of the weather data we extracted.
I hope to pick four series to model on.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from dataprep.eda import create_report

# =============================================================================
# FUNCTIONS
# =============================================================================
def plotSeries(df, col):
    """creates time series plots"""
    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['datetime'], df[col], label=col)
    ax.set_xlabel('Time')
    ax.set_ylabel(col)
    ax.set_title(col.upper())
    ax.grid(True)
    ax.legend(loc='upper left')

# =============================================================================
# PROCESS 
# =============================================================================
#load data
df = pd.read_csv('Data/weather_data.csv')

#quick look at the columns
df.head()

"""
We only need one of the datetime columns.
We'll consider the value of each series once we hav plotted them.
Sunrise/Sunset is of no interest to this project.
The last 3 columns are of no use either
"""

#drop unneeded columns
cols_drop = ['datetimeEpoch', 'sunrise', 'sunriseEpoch', 'sunset',\
             'sunsetEpoch', 'stations', 'source', 'tzoffset']
    
df = df.drop(cols_drop, axis=1)
    
#check data types
df.info()

"""
datetime may benefit from being in datetime format
We can remove columns with only null values
"""

#drop unneeded columns pt.2
cols_drop = ['precipprob', 'preciptype']
    
df = df.drop(cols_drop, axis=1)

#convert to datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y')

#summary stats
df.describe()
"""
Since this is a time series problem, I'm not really looking for quite the same things as if it were a regression problem e.g. skewness
Interesting to see that the variability differs between the 3 temp metrics.
I dont see any extreme min or max values that would indicate data anomalies.
See nothing for Snow which cant be right given i live in the area and distinctly remember plenty of snow over the years. We'll drop these.
"""

#drop unneeded columns pt.3
cols_drop = ['snow', 'snowdepth']
    
df = df.drop(cols_drop, axis=1)

#auto eda (dataprep library)
"""I Like pandas profiling library but lets try an alternative for this EDA"""
create_report(df, title='Weather_data_EDA')

"""
tempmax is skewed unlike other temps
precip shows that there is no rain one in three days. this is confirmed by the conditions columns. Love London!
winddir showing that the prevaling wind comes from the Atlantic as expected.
cloudcover showing that the will always be some clouds hanging about. Rare for it to be over 30% though.
Interesting inverse correlation between humidty and solar radiation. Otherwise the correlations amongst the other variables come at no surprise..
"""

#time series plots
for col in df.columns:
    plotSeries(df, col)

"""
Temperature behaving in accordance with seasons/time of year as expected.
Humidity is the inverse of temperature but more noisy.
Patterns in precipitation are not so easy to discern.
Windspeed is mostly noise centered around a mean of 13.
Strange jump in cloudcover at the end of 2020. Maybe they changed how it was measured? I wouldve liked to have modelled this but now im not so sure.

The four variables that will be modelled are: temp, humidity, precip, windspeed
"""