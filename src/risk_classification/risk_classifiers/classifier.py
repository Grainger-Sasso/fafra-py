from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.validation.cross_validator import CrossValidator


class Classifier(ABC):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.scaler: StandardScaler = StandardScaler()
        self.cross_validator = CrossValidator()
        self.params = {}

    def get_params(self):
        return self.params

    def get_name(self):
        return self.name

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

    def split_input_metrics(self, input_metrics: InputMetrics,
                            test_size=0.33, random_state=42):
        # Input to test_train_split shape is (n_samples, n_features)
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def scale_input_data(self, input_metrics: InputMetrics) -> InputMetrics:
        scaled_input_metrics = InputMetrics()
        metrics, names = input_metrics.get_metric_matrix()
        # Shape of fit input is (n_samples, n_features)
        self.scaler.fit(metrics)
        # Shape of transform input is (n_samples, n_features)
        scaled_metrics = self.scaler.transform(metrics)
        # Convert output shape of transform from (n_samples, n_features) to
        # (n_features, n_samples)
        scaled_metrics = scaled_metrics.T
        for name, metric in zip(names, scaled_metrics):
            input_metric = InputMetric(name, metric)
            scaled_input_metrics.set_metric(name, input_metric)
        scaled_input_metrics.set_labels(input_metrics.get_labels())
        return scaled_input_metrics

    def scale_train_test_data(self, x_train, x_test):
        # Fit the scaler to the training data
        self.scaler.fit(x_train)
        # Transform the training data
        x_train_t = self.scaler.transform(x_train)
        # Transform the test data
        x_test_t = self.scaler.transform(x_test)
        return x_train_t, x_test_t

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred, output_dict=True)

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
