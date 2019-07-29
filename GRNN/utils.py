
from sklearn.model_selection import train_test_split
from sklearn import datasets, preprocessing
from neupy import algorithms
import pandas as pd
import numpy as np
import os, sys

def read_dataset(_path = 'cleaned_events_weather.csv'):
	""" reads data from pandas in dataframe"""
	path = os.path.abspath(os.path.join("GRNN", _path))
	df = pd.read_csv(path)
	# copy dataFrame for future use
	df_copy = df.copy(deep=True)
	df = df.replace('?', 0)
	df.replace(np.nan, 0, inplace=True)
	# df.fillna(0)
	return df, df_copy


def get_features_max_temp(data):
	""" returns X and Y with desired features """
	data = data[['Mean TemperatureC', 'Precipitationmm', 'Min TemperatureC', 'Dew PointC', 'MeanDew PointC',\
	'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity','Max VisibilityKm', 'Mean VisibilityKm',\
	'Min VisibilitykM', 'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'CloudCover','WindDirDegrees', 'Max Wind SpeedKm/h', 'Max TemperatureC']]

	X = data.iloc[:, 2:-1].values
	Y = data.iloc[:, -1:].values
	return X, Y

def get_features_min_temp(data):
	""" returns X and Y with desired features """
	data = data[['Mean TemperatureC', 'Precipitationmm', 'Max TemperatureC', 'Dew PointC', 'MeanDew PointC',\
	'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity','Max VisibilityKm', 'Mean VisibilityKm',\
	'Min VisibilitykM', 'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'CloudCover','WindDirDegrees', 'Max Wind SpeedKm/h', 'Min TemperatureC']]

	X = data.iloc[:, 2:-1].values
	Y = data.iloc[:, -1:].values
	return X, Y


def get_features_wind_speed(data):
	""" returns X and Y with desired features """
	data = data[['Mean TemperatureC', 'Precipitationmm', 'Min TemperatureC', 'Dew PointC', 'MeanDew PointC',\
	'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity','Max VisibilityKm', 'Mean VisibilityKm',\
	'Min VisibilitykM', 'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'CloudCover','WindDirDegrees', 'Max TemperatureC', 'Max Wind SpeedKm/h']]

	X = data.iloc[:, 2:-1].values
	Y = data.iloc[:, -1:].values
	return X, Y

def get_features_percipitation(data):
	""" returns X and Y with desired features """
	data = data[['Mean TemperatureC', 'Max Wind SpeedKm/h', 'Min TemperatureC', 'Dew PointC', 'MeanDew PointC',\
	'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity','Max VisibilityKm', 'Mean VisibilityKm',\
	'Min VisibilitykM', 'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'CloudCover','WindDirDegrees', 'Max TemperatureC', 'Precipitationmm']]

	X = data.iloc[:, 2:-1].values
	Y = data.iloc[:, -1:].values
	return X, Y


def split_dataset(X, Y, test_ratio = 0.3):
	""" split dataset into two training and testing """

	return train_test_split(
    preprocessing.minmax_scale(X),
    preprocessing.minmax_scale(Y.reshape((-1, 1))),
    test_size=test_ratio,
	)

def train_GRNN(x_train, y_train):
	""" returns the trained GRNN """
	nw = algorithms.GRNN(std=0.1, verbose=False)
	nw.train(x_train, y_train)
	return nw

def convert_events_to_numeric(events_list):

	# keeps track of events2index
	unique_events = set(events_list)
	events2index = {e:i for i,e in enumerate(unique_events)}

	# print(events2index)
	mapped_events_list = [] # numeric mappings

	for i, x in enumerate(events_list):
		mapped_events_list.append(events2index[x])

	return mapped_events_list





