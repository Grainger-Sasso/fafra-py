from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import lightgbm as lgb
import optuna


class LightGBMRiskClassifier:
    # Actual preprocessing (e.g., scaling) will be done using the metrics that Grainger and Dr. Hernandez
    # have developed.
    # Also, should eventually use 5-fold cross-validation
    def __init__(self, params: dict):
        # params is a dict of parameters, including hyperparameters, for the LightGBM risk classifier
        self.model = lgb.LGBMClassifier(params)
        self.scaler = preprocessing.StandardScaler()

    def get_model(self) -> lgb.LGBMClassifier():
        return self.model

    def set_model(self, params: dict):
        self.model = lgb.LGBMClassifier(params)

    def get_scaler(self) -> preprocessing.StandardScaler():
        return self.scaler

    def set_scaler(self, scaler=preprocessing.StandardScaler()):
        self.scaler = scaler

    def scale_input_data(self, x):
        self.scaler.fit(x)
        return self.scaler.transform(x)

    def scale_train_test_data(self, x_train, x_test):
        # Fit the scaler to the training data
        self.scaler.fit(x_train)
        # Transform the training data
        x_train_t = self.scaler.transform(x_train)
        # Transform the test data
        x_test_t = self.scaler.transform(x_test)
        return x_train_t, x_test_t

    def split_input_metrics(self, x, y, test_size=0.3, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def fit_model(self, x: np.ndarray, y: np.ndarray):
        """
        Fits LightGBM model to input training vectors, x, and target values, y (notation is canonically used)
        :param x: Training vectors
        :type: np.ndarray
        :param y: Target values
        :type: np.ndarray
        :return: Trained model
        :rtype: lgb.LGBMClassifier()
        """
        self.model.fit(x, y)

    def make_prediction(self, samples):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    # must implement k-fold cross validation in code (not needed for first dataset, which has only 340 examples)


# use numpy to read csv data and then choose values of hyperparameters to create good LightGBM risk classifier object
x = np.genfromtxt('C:\\Users\\fancy\\Downloads\\x_data_metrics.csv', delimiter=',')
y = np.genfromtxt('C:\\Users\\fancy\\Downloads\\y_data_metrics.csv', delimiter=',')
