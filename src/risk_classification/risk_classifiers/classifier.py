from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.risk_classification.validation.cross_validator import CrossValidator


class Classifier(ABC):
    def __init__(self, model):
        self.model = model
        self.scaler: StandardScaler = StandardScaler()
        self.cross_validator = CrossValidator()

    def get_model(self):
        return self.model

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def get_cross_validator(self) -> CrossValidator:
        return self.cross_validator

    def set_model(self, model):
        self.model = model

    def set_scaler(self, scaler):
        self.scaler = scaler

    def split_input_metrics(self, x, y, test_size=0.33, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

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

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    @abstractmethod
    def train_model(self, x, y, **kwargs):
        pass

    @abstractmethod
    def make_prediction(self, samples, **kwargs):
        pass

    @abstractmethod
    def score_model(self, x_test, y_test, **kwargs):
        pass

    @abstractmethod
    def cross_validate(self, x, y, folds=5, **kwargs):
        pass
