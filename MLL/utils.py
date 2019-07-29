import pandas as pd
import numpy as np
import os

def read_dataset(_path = 'cleaned_events_weather.csv'):
	""" read dataset from disk """
	path = os.path.abspath(os.path.join("MLL", _path))

	# print("Reading dataset from file: {}".format(path))
	df = pd.read_csv(path)
	# copy dataFrame for future use
	df_copy = df.copy(deep=True)
	df = df.replace('?', 0)
	df.replace(np.nan, 0, inplace=True)
	# df.fillna(0)
	return df, df_copy

def convert_events_to_numeric(events_list):

	# keeps track of events2index
	unique_events = set(events_list)
	events2index = {e:i for i,e in enumerate(unique_events)}

	# print(events2index)
	mapped_events_list = [] # numeric mappings

	for i, x in enumerate(events_list):
		mapped_events_list.append(events2index[x])

	return mapped_events_list


# extract features from dataset - cleaned_events_weather.csv
# maxTemp, meanTemp, minTemp, dew = data["Max TemperatureC"].values, data["Mean TemperatureC"].values, data["Min TemperatureC"].values, data["Dew PointC"].values
# meanDew, minDew, maxHumidity = data["MeanDew PointC"].values, data["Min DewpointC"].values, data["Max Humidity"].values
# meanHumidity, minHumidity, maxSeaLevel = data["Mean Humidity"].values, data["Min Humidity"].values, data["Max Sea Level PressurehPa"].values
# meanSeaLevel, minSeaLevel = data["Mean Sea Level PressurehPa"].values, data["Min Sea Level PressurehPa"].values
# maxVisibility, meanVisibility, minVisibility  = data["Max VisibilityKm"].values, data["Mean VisibilityKm"].values, data["Min VisibilitykM"].values
# maxWind, meanWind, maxGust = data["Max Wind SpeedKm/h"].values, data["Mean Wind SpeedKm/h"].values, data["Max Gust SpeedKm/h"].values
# perceip, cloudCover, events, windDir = data["Precipitationmm"].values, data["CloudCover"].values, data["Events"].values, data["WindDirDegrees"].values
# perceip, events = data["Precipitationmm"].values, data["Events"].values # austin_weather.csv

def get_x0(data):
	
	maxTemp = data["Max TemperatureC"].values
	m = len(maxTemp)
	x0 = np.ones(m)

	return x0

def max_temp_features(data):
	""" get X and Y for max/min Temperature predictions """

	meanTemp, meanHumidity = data["Mean TemperatureC"].values, data["Mean Humidity"].values,
	maxTemp, dew = data["Max TemperatureC"].values,  data["Dew PointC"].values # predictant = maxTemp

	X = np.array([get_x0(data), meanTemp, meanHumidity, dew])
	X = X.T  # transpose
	Y = np.array(maxTemp) # minTemp to predict minTemperature
	return X, Y

def min_temp_features(data):
	""" get X and Y for max/min Temperature predictions """

	meanTemp, meanHumidity = data["Mean TemperatureC"].values, data["Mean Humidity"].values,
	minTemp, dew = data["Min TemperatureC"].values,  data["Dew PointC"].values # predictant = minTemp

	X = np.array([get_x0(data), meanTemp, meanHumidity, dew])
	X = X.T
	Y = np.array(minTemp) # minTemp to predict minTemperature
	return X, Y

def percipitation_features(data):
	""" get X and Y for percipitation predictions """
	# events_numeric = np.array(convert_events_to_numeric(events))

	meanDew, meanHumidity = data["MeanDew PointC"].values, data["Mean Humidity"].values
	minVisibility, meanVisibility = data["Min VisibilitykM"].values, data["Mean VisibilityKm"].values
	maxVisibility, cloudCover = data["Max VisibilityKm"].values, data["CloudCover"].values
	# predictant
	percip = data["Precipitationmm"].values
	X = np.array([get_x0(data), meanDew, meanHumidity, minVisibility, meanVisibility,\
					maxVisibility, cloudCover])
	X = X.T
	Y = np.array(percip) # to predict Percipitation

	# print("percipitation_features")
	# print(X.shape)
	# print(Y.shape)
	return X, Y

def max_wind_features(data):
	""" get X and Y for max Wind Speed predictions """
	# events_numeric = np.array(convert_events_to_numeric(events))

	meanWind, minTemp = data["Mean Wind SpeedKm/h"].values, data["Min TemperatureC"].values
	minSeaLevel, maxHumidity = data["Min Sea Level PressurehPa"].values, data["Max Humidity"].values
	# predictant
	maxWind = data["Max Wind SpeedKm/h"].values
	X = np.array([get_x0(data), meanWind, minTemp, minSeaLevel, maxHumidity])
	X = X.T
	Y = np.array(maxWind) # to predict minTemperature
	return X, Y