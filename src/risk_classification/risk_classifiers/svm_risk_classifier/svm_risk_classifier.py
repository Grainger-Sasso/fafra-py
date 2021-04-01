import numpy as np
import cvxopt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


class SVMRiskClassifier:
    def __init__(self):
        self.model = None

    def get_model(self):
        return self.model

    def set_model(self, model: SVC):
        self.model = model

    def generate_model(self, kernel='linear', C=1E10):
        # TODO: come back and fix the input params for model
        self.set_model(SVC(kernel=kernel, C=C))

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
