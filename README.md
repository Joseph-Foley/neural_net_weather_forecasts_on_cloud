# neural_net_weather_forecasts_on_cloud

## Contents

1.	[Intro](#Intro)
2.	[Repo Structure])(#Repo Structure)
3.	[Results](#Results)
4.	[Further Improvements](#Further Improvements)

## Intro
This Repo shows how to take an API feed of weather data and train a recurrent neural network in order to provide a weather forecast. Predictions will be deployed and visualised on heroku cloud platform. Forecasts update daily.

## Repo Structure
There are 2 main segments to this Repo.

If you want to follow along with the analysis & modelling, please go to the “Notebooks” directory. The Jupyter notebooks are numbered in a logical order that makes it easy to follow along. The first shows an exploratory analysis of weather API data taken from visualcrossing.com. The second notebook establishes a baseline weather forecasting model that is to be beaten in the main modelling scripts. In the third notebook I make a class using Tensorflow that creates LSTM or GRU networks to forecast the weather, the results were poor. The fourth notebook addresses the poor results by introducing Gaussian smoothing factors into the main model.

Otherwise the main event is in the script **dash_app.py** that can be seen in the primary directory. This script calls the weather API for historical data. It then imports the models & classes created in the analysis segment to provide a 7 day forecast of temperature, humidity, precipitation and wind speed. The historical data and forecasts are then passed into a dashboard made using the Dash & Plotly libraries. This can be hosted locally or deployed on a cloud service. 

This repo is currently being hosted on Heroku cloud service. The resulting dashboard can be viewed here: [link](nn-cloud.herokuapp.com "Title")

## Results
The baseline model of projecting the last observation forward had a **mean absolute error of 1.7** (temperature – degrees centigrade). The final model had an error of **0.7**.

The weather forecasts can be viewed at the link above and should resemble the image below.
![Alt](Plotly/ nn-cloud.png "Title")

## Further Improvements
- Model max temperature instead of temp. It’s more typical of what we perceive when thinking about how hot or cold the day will be.
- Experiment with the time series generator to set X to be the Gaussian smoothed series but y to be the unsmoothed series.
- Input multiple weather series into one model as its possible that one weather variable may have an impact on another.
- Experiment with models that predict more than one step ahead so that extending a forecast doesn’t require a prediction to be made off of another prediction.
- Allow user input on the Dashboard so that they can generate as many days forecast as desired.	
