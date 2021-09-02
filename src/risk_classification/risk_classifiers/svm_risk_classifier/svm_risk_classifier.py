import numpy as np
import cvxopt
import os
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


class SVMRiskClassifier:
    def __init__(self, kernel='linear', c=1E10):
        self.model: SVC = SVC(kernel=kernel, C=c)
        self.scaler: StandardScaler = preprocessing.StandardScaler()

    def get_model(self) -> SVC:
        return self.model

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def set_model(self, model: SVC):
        self.model = model

    def set_scaler(self, scaler: StandardScaler):
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

    def split_input_metrics(self, x, y, test_size=0.33, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def fit_model(self, x: np.ndarray, y: np.ndarray):
        """
        Fits SVM model to input training vectors, x, and target values, y (notation is canonically used)
        :param x: Training vectors
        :type: np.ndarray
        :param y: Target values
        :type: np.ndarray
        :return: Trained model
        :rtype: SVC
        """
        self.model.fit(x, y)

    def make_prediction(self, samples):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)


def main():
    classifier = SVMRiskClassifier()
    #read data file
    rootPath = 'D:/carapace/metric_test_data'
    train_x_path = os.path.join(rootPath, 'x_data_metrics.csv')
    train_y_path = os.path.join(rootPath, 'y_data_metrics.csv')
    train_x = pd.read_csv(train_x_path, delimiter=',', header=None)
    train_y = pd.read_csv(train_y_path, delimiter=',', header=None)
    scaler = classifier.get_scaler()
    x_data_scaled = scaler.fit_transform(train_x)
    x_train, x_test, y_train, y_test = classifier.split_input_metrics(x_data_scaled, train_y.values.ravel())
    print(x_train,file=open("./x_train.txt",'a'))
    classifier.fit_model(x_train, y_train)
    ypred = classifier.make_prediction(x_test)
    #create confusion matrix and see the classification report
    result = confusion_matrix(y_test, ypred)
    print("Confusion Matrix is {}".format(result))
    report = classifier.create_classification_report(y_test, ypred)
    print(report)
    print("Accuracy: ", classifier.score_model(x_test, y_test))



if __name__ ==  '__main__':
    main()
