import numpy as np
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
