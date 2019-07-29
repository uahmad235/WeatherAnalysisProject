
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import os


def read_dataset(_path = 'cleaned_events_weather.csv'):

    # join path to read dataset from
    path = os.path.abspath(os.path.join("KNN",_path))
    # read dataset using pandas
    df = pd.read_csv(path)
    # copy dataFrame for future use
    df_copy = df.copy(deep=True)
    # replace null or unknown values with 0
    df = df.replace('?', 0)
    df.replace(np.nan, 0, inplace=True)
    # df.fillna(0)
    return df, df_copy

# extract features from dataset
# maxTemp, meanTemp, minTemp, dew = data["Max TemperatureC"].values, data["Mean TemperatureC"].values, data["Min TemperatureC"].values, data["Dew PointC"].values
# meanDew, minDew, maxHumidity = data["MeanDew PointC"].values, data["Min DewpointC"].values, data["Max Humidity"].values
# meanHumidity, minHumidity, maxSeaLevel = data["Mean Humidity"].values, data["Min Humidity"].values, data["Max Sea Level PressurehPa"].values
# meanSeaLevel, minSeaLevel = data["Mean Sea Level PressurehPa"].values, data["Min Sea Level PressurehPa"].values
# maxVisibility, meanVisibility, minVisibility  = data["Max VisibilityKm"].values, data["Mean VisibilityKm"].values, data["Min VisibilitykM"].values
# maxWind, meanWind, maxGust = data["Max Wind SpeedKm/h"].values, data["Mean Wind SpeedKm/h"].values, data["Max Gust SpeedKm/h"].values
# perceip, cloudCover, events, windDir = data["Precipitationmm"].values, data["CloudCover"].values, data["Events"].values, data["WindDirDegrees"].values


# for prediction of MinTemp k = 15(95%)
def min_temp_features(data):
    
    data = data[['Mean TemperatureC', 'Max TemperatureC', 'Dew PointC', 'MeanDew PointC',\
     'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity', 'Max Sea Level PressurehPa','Mean Sea Level PressurehPa',\
     'Min Sea Level PressurehPa','Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h', 'Min TemperatureC']]
    return data

# for prediction of MaxTemp k = 31, 19, 15 = 95.3952(MaxTemp)
def max_temp_features(data):

    data = data[['Mean TemperatureC', 'Dew PointC', 'MeanDew PointC',\
     'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity', 'Max Sea Level PressurehPa',\
     'Mean Sea Level PressurehPa','Min Sea Level PressurehPa',\
      'Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h', 'Min TemperatureC', 'Max TemperatureC']]
    return data

# for prediction of MaxWind k=15(65.32%)
def max_wind_features(data):

    data = data[['Dew PointC', 'Max Humidity', 'Mean Humidity', 'Max Sea Level PressurehPa',\
    'Mean Sea Level PressurehPa','Min Sea Level PressurehPa',\
     'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'WindDirDegrees', 'Max Wind SpeedKm/h']]
    return data

# percipitation prediction #5, 31, 15, 9, 19=>36% acc. (highest) percp
def precipitation_features(data):

    data = data[['Mean TemperatureC', 'Min TemperatureC', 'Dew PointC', 'MeanDew PointC',\
     'Min DewpointC','Max Humidity', 'Mean Humidity', 'Min Humidity','Max VisibilityKm', 'Mean VisibilityKm',\
     'Min VisibilitykM', 'Mean Wind SpeedKm/h', 'Max Gust SpeedKm/h', 'CloudCover','WindDirDegrees', 'Max Wind SpeedKm/h', 'Precipitationmm']]

    return data

def convert_events_to_numeric(events_list):

	# keeps track of events2index
	unique_events = set(events_list)
	events2index = {e:i for i,e in enumerate(unique_events)}

	# print(events2index)
	mapped_events_list = [] # numeric mappings

	for i, x in enumerate(events_list):
		mapped_events_list.append(events2index[x])

	return mapped_events_list