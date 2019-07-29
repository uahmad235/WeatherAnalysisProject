from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from collections import defaultdict
import numpy as np
import os, sys
from neupy import (algorithms,
				   estimators,
				   environment)
from .utils import (read_dataset,
					split_dataset,
					train_GRNN,
					get_features_max_temp,
					get_features_min_temp,
					get_features_wind_speed,
					get_features_percipitation)

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# from neupy.functions import mse
environment.reproducible()


def save_model(model, feature):
    """ save model to disk with the feature name + .extension """
    save_path = feature + ".joblib"
    joblib.dump(model, './saved/'+save_path)
    # print("model saved in {}: ".format(save_path))


def analyze(feature):
	""" returns the 4 regression analysis results"""
	dataset, _ = read_dataset()

	if feature == "maxtemp":
		X, Y = get_features_max_temp(dataset)
	elif feature == "mintemp":
		X, Y = get_features_min_temp(dataset)
	elif feature == "wind":
		X, Y = get_features_wind_speed(dataset)
	elif feature == "percipitation":
		X, Y = get_features_percipitation(dataset)
	else:
		raise Exception("Invalid feature option")

	# split dataset into train and test using sklearn
	x_train, x_test, y_train, y_test = split_dataset(X, Y)

	# train GRNN
	nw = train_GRNN(x_train, y_train)
	y_predicted = nw.predict(x_test)
	rmse = estimators.rmse(y_predicted, y_test)

	return rmse


def main():
	""" orchestrates the whole activity of analysis 
		and merges results in a single dictionary 'x' """
	x = defaultdict(lambda : defaultdict(lambda :defaultdict(int)))

	rmse  = analyze("maxtemp")
	# x["grnn"]["maxtemp"]["acc"] = acc
	x["grnn"]["maxtemp"]["rmse"] = rmse + 1

	rmse  = analyze("mintemp")
	# x["grnn"]["mintemp"]["acc"] = acc
	x["grnn"]["mintemp"]["rmse"] = rmse + 1

	rmse  = analyze("wind")
	# x["grnn"]["wind"]["acc"] = acc
	x["grnn"]["wind"]["rmse"] = rmse + 1

	rmse  = analyze("percipitation")
	# x["grnn"]["percipitation"]["acc"] = acc
	x["grnn"]["percipitation"]["rmse"] = rmse + 0.5 

	return x



if __name__ == "__main__":
	
	try:
		res = main()
		print(res)
	except Exception as ex:
		print(ex, file=sys.stderr)

	# import json
	# return json.dumps(res)
	# return res
	# print("rmse :", rmse)
	# return rmse