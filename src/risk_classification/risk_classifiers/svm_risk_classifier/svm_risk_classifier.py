import numpy as np
import cvxopt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


class SVMRiskClassifier:
    def __init__(self, kernel='linear', c=1E10):
        self.model: SVC = SVC(kernel=kernel, C=c)
        self.scaler: StandardScaler = preprocessing.StandardScaler()

    def get_model(self) -> SVC:
        return self.model

    def set_model(self, model: SVC):
        self.model = model

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def set_scaler(self, scaler: StandardScaler):
        self.scaler = scaler

    def fit_scaler(self, x_train):
        self.scaler.fit(x_train)

    def transform_data(self, x):
        return self.scaler.transform(x)

    def split_input_metrics(self, x, y, test_size=0.33, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def cross_val_model(self, x, y, cv=None, return_estimator=True):
        # cv_results: [dict]; "test_score", "train_score", "fit_time", "score_time", "estimator"
        cv_results = cross_validate(self.get_model(), x, y, cv=cv, return_estimator=return_estimator)
        if return_estimator:
            self.set_model(cv_results['estimator'])
            cv_results = {
                'test_score': cv_results['test_score'],
                'train_score': cv_results['train_score'],
                'fit_time': cv_results['fit_time']
            }
        return cv_results

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
