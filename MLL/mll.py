from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .utils import (read_dataset,
					max_temp_features,
					min_temp_features,
					max_wind_features,
					percipitation_features)
import sys, os
from collections import defaultdict
from sklearn.externals import joblib
# import faulthandler
# faulthandler.enable()
# df["weight"].mean()  -- mean value of a dataframe

def save_model(regressor, feature):
    """ save model to disk with the feature name + .extension """
    # save_path = feature + ".joblib"
    save_path = os.path.abspath(os.path.join("MLL", "saved", feature+'.joblib'))
    # print("saving to  path:" , save_path)

    joblib.dump(regressor, save_path)
    # print("model saved in {}: ".format(save_path))


def load_model(feature):
	""" loads model from the disk against specified feature """
	# load_path = './saved/'+feature+'.joblib'
	load_path = os.path.abspath(os.path.join("MLL", "saved", feature+'.joblib'))
	regressor = joblib.load(load_path)
	return regressor

def build_model():
	pass

def analyze(feature):

	data, df_copy = read_dataset() # path = 'cleaned_events_dataset.csv'

	if feature == "maxtemp":
		X, Y = max_temp_features(data)
		# print(feature)
	elif feature == "mintemp":
		X, Y = min_temp_features(data)
		# print(feature)
	elif feature == "wind":
		X, Y = max_wind_features(data)
		# print(feature)
	elif feature == "percipitation":
		X, Y = percipitation_features(data)
		# print(feature)
	else:
		raise Exception("Unknown Option Selected as feature")

	# print("xshape", X.shape)
	# print(Y.shape)

	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

	# Model Intialization
	# reg = LinearRegression()  # args: fit_intercept=True, normalize=True
	# model training
	# reg = reg.fit(train_X, train_Y)
	# save_model(reg, feature)
	
	# print("mll ")
	# predictions
	reg = load_model(feature)
	Y_pred = reg.predict(test_X)

	# Model Evaluation
	rmse = np.sqrt(mean_squared_error(test_Y, Y_pred))
	r2 = reg.score(test_X, test_Y)

	return r2, rmse


def main():

	x = defaultdict(lambda : defaultdict(lambda :defaultdict(int)))

	acc, rmse  = analyze("maxtemp")
	x["mll"]["maxtemp"]["acc"] = acc * 100
	x["mll"]["maxtemp"]["rmse"] = rmse

	acc, rmse  = analyze("mintemp")
	x["mll"]["mintemp"]["acc"] = acc * 100
	x["mll"]["mintemp"]["rmse"] = rmse

	acc, rmse  = analyze("wind")
	x["mll"]["wind"]["acc"] = acc * 100
	x["mll"]["wind"]["rmse"] = rmse

	acc, rmse  = analyze("percipitation")
	x["mll"]["percipitation"]["acc"] = acc * 100
	x["mll"]["percipitation"]["rmse"] = rmse

	return x


if __name__ == "__main__":

	try:
		print(main())
	except Exception as ex:
		print(ex, file=sys.stderr)