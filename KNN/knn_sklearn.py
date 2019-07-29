
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             mean_squared_error,
                             accuracy_score) 
from .utils import (read_dataset,
                   max_temp_features,
                   min_temp_features,
                   max_wind_features,
                   precipitation_features)
from math import sqrt
from sklearn.externals import joblib
from collections import defaultdict
import json, sys, os

# utils.multiclass.type_of_target(y_test) # args: y_testz, y_train

def scale_data(X_train, X_test, y_train):
    """ returns normalizd X and y values for improved performance """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.ravel()

    return X_train, X_test, y_train

def prepare_data(data):
    """ splits and scales data for input """
    
    X = data.iloc[:, 2:-1].values
    Y = data.iloc[:, -1:].values

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                         test_size=0.33,
                                                         random_state=42)

    # scale data for KNN improved accuracy
    X_train, X_test, y_train = scale_data(X_train, X_test, y_train)
    return  X_train, X_test, y_train, y_test

def train(X_train, y_train, K=15):
    """ returns MLR model trained on X_train and y_train """
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  
    regressor = KNeighborsRegressor(n_neighbors=K)
    regressor.fit(X_train, y_train)
    
    return regressor

def make_predictions(regressor, X_test):
    """ returns predictios on X_test"""
    y_pred = regressor.predict(X_test)
    return y_pred

def evaluate(y_test, y_pred):
    """ returns mse (mean squared error) on prediction vs actual """
    from sklearn import preprocessing
    lab_enc = preprocessing.LabelEncoder()
    y_test_encoded = lab_enc.fit_transform(y_test.ravel())
    y_pred_encoded = lab_enc.fit_transform(y_pred)    

    return np.sqrt(mean_squared_error(y_test, y_pred))

def save_model(regressor, feature):
    """ save model to disk with the feature name + .extension """
    # save_path = feature + ".joblib"
    save_path = os.path.abspath(os.path.join("KNN", "saved", feature+'.joblib'))
    joblib.dump(regressor, save_path)

    # uncoment this to save model from node.js app
    # path = os.path.abspath(os.path.join("KNN", "saved", _path))
    # print("model saved in {}: ".format(save_path))

def analyze(feature):
    """ orchestrates the whole activity of analysis and returns
        feature, mse and accuracy """
    # read dataset from utils.py
    data, _ = read_dataset()
    # checks the feature to be analyzed
    if feature == "maxtemp":
        data = max_temp_features(data)
    elif feature == "mintemp":    
        data = min_temp_features(data)
    elif feature == "wind":
        data = max_wind_features(data)
    elif feature == "percipitation":
        data = precipitation_features(data)
    else:
        raise Exception("Unknown Option Selected as feature")

    X_train, X_test, y_train, y_test = prepare_data(data)

    #  uncomment below line to train model
    # regressor = train(X_train, y_train)
    # load_path = './saved/'+feature+'.joblib'
    # load model from disk
    load_path = os.path.abspath(os.path.join("KNN", "saved", feature+'.joblib'))
    regressor = joblib.load(load_path)

    # mean square error
    y_pred = make_predictions(regressor, X_test)
    mse = evaluate(y_test, y_pred)

    # calculate accuracy score
    acc = regressor.score(X_test, y_test)

    # print("KNN validation Accuracy of feature {} is {}: ".format(feature, acc))
    save_model(regressor, feature)

    return feature, acc, mse 


def main():
    """ returns the combined response of all 4 features """
    x = defaultdict(lambda : defaultdict(lambda :defaultdict(int)))

    feature, acc, mse  = analyze("maxtemp")
    x["knn"]["maxtemp"]["acc"] = acc * 100
    x["knn"]["maxtemp"]["mse"] = mse

    feature, acc, mse  = analyze("mintemp")
    x["knn"]["mintemp"]["acc"] = acc * 100
    x["knn"]["mintemp"]["mse"] = mse

    feature, acc, mse  = analyze("wind")
    x["knn"]["wind"]["acc"] = acc * 100
    x["knn"]["wind"]["mse"] = mse    

    feature, acc, mse  = analyze("percipitation")
    x["knn"]["percipitation"]["acc"] = acc * 100
    x["knn"]["percipitation"]["mse"] = mse

    return x


if __name__ == "__main__":
    
    
    try:
        print(main())
    except Exception as ex:
        print(ex, file=sys.stderr)
        # print(ex)