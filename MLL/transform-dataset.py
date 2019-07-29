
import pandas as pd
import numpy as np
from utils import read_dataset
from utils import convert_events_to_numeric


# def transform_dataset(data, df_copy):

# df = no-nan , df_copy = nan
data, df_copy  = read_dataset()

# extract features from dataset
maxTemp, meanTemp, minTemp, dew = data["Max TemperatureC"].values, data["Mean TemperatureC"].values, data["Min TemperatureC"].values, data["Dew PointC"].values
meanDew, minDew, maxHumidity = data["MeanDew PointC"].values, data["Min DewpointC"].values, data["Max Humidity"].values
meanHumidity, minHumidity, maxSeaLevel = data[" Mean Humidity"].values, data[" Min Humidity"].values, data[" Max Sea Level PressurehPa"].values
meanSeaLevel, minSeaLevel = data[" Mean Sea Level PressurehPa"].values, data[" Min Sea Level PressurehPa"].values
maxVisibility, meanVisibility, minVisibility  = data[" Max VisibilityKm"].values, data[" Mean VisibilityKm"].values, data[" Min VisibilitykM"].values
maxWind, meanWind, maxGust = data[" Max Wind SpeedKm/h"].values, data[" Mean Wind SpeedKm/h"].values, data[" Max Gust SpeedKm/h"].values
perceip, cloudCover, events, windDir = data["Precipitationmm"].values, data[" CloudCover"].values, data[" Events"].values, data["WindDirDegrees"].values

# removes NaN from dataset
meanMaxTemp, meanMeanTemp, meanMinTemp, meanDew = np.mean(maxTemp), np.mean(meanTemp), np.mean(minTemp), np.mean(dew)
df_copy["Max TemperatureC"].fillna(meanMaxTemp, inplace=True)
df_copy["Mean TemperatureC"].fillna(meanMeanTemp, inplace=True)
df_copy["Min TemperatureC"].fillna(meanMinTemp, inplace=True)
df_copy["Dew PointC"].fillna(meanDew, inplace=True)

meanMeanDew, meanMinDew, meanMaxHumidity = np.mean(meanDew), np.mean(minDew), np.mean(maxHumidity)
df_copy["MeanDew PointC"].fillna(meanMeanDew, inplace=True)
df_copy["Min DewpointC"].fillna(meanMinDew, inplace=True)
df_copy["Max Humidity"].fillna(meanMaxHumidity, inplace=True)

meanMeanHumidity, meanMinHumidity, meanMaxSeaLevel = np.mean(meanHumidity), np.mean(minHumidity), np.mean(maxSeaLevel)
df_copy["Mean Humidity"].fillna(meanMeanHumidity, inplace=True)
df_copy["Min Humidity"].fillna(meanMinHumidity, inplace=True)
df_copy["Max Sea Level PressurehPa"].fillna(meanMaxSeaLevel, inplace=True)

meanMeanSeaLevel, meanMinSeaLevel = np.mean(meanSeaLevel), np.mean(minSeaLevel)
df_copy["Mean Sea Level PressurehPa"].fillna(meanMeanSeaLevel, inplace=True)
df_copy["Min Sea Level PressurehPa"].fillna(meanMinSeaLevel, inplace=True)


meanMaxVisibility, meanMeanVisibility, meanMinVisibility  = np.mean(maxVisibility), np.mean(meanVisibility), np.mean(minVisibility)
df_copy["Max VisibilityKm"].fillna(meanMaxVisibility, inplace=True)
df_copy["Mean VisibilityKm"].fillna(meanMeanVisibility, inplace=True)
df_copy["Min VisibilitykM"].fillna(meanMinVisibility, inplace=True)

meanMaxWind, meanMeanWind, meanMaxGust = np.mean(maxWind), np.mean(meanWind), np.mean(maxGust)
df_copy["Max Wind SpeedKm/h"].fillna(meanMaxWind, inplace=True)
df_copy["Mean Wind SpeedKm/h"].fillna(meanMeanWind, inplace=True)
df_copy["Max Gust SpeedKm/h"].fillna(meanMaxGust, inplace=True)

meanPerceip, meanCloudCover, meanWindDir = np.mean(perceip), np.mean(cloudCover), np.mean(windDir)
df_copy["Precipitationmm"].fillna(meanPerceip, inplace=True)
df_copy["CloudCover"].fillna(meanCloudCover, inplace=True)
df_copy["WindDirDegrees"].fillna(meanWindDir, inplace=True)

# df_copy.to_csv('cleaned_weather_dataset.csv', sep=',')


# mapped_events = convert_events_to_numeric(events)

# n = df_copy.columns[21]
# # Drop that column
# df_copy.drop(n, axis = 1, inplace = True)

# df_copy[n] = mapped_events

# df_copy.to_csv('cleaned_events_weather.csv', sep=',')
# read_csv_custom = pd.read_csv('cleaned_events_weather.csv')

# print(len(df_copy.columns))
# print(len(read_csv_custom.columns))

# print(n)