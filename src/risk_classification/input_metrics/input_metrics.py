import numpy as np
from typing import Dict


from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.dataset_tools.risk_assessment_data.user_data import UserData


class InputMetrics:
    def __init__(self):
        self.metrics: Dict[MetricNames: InputMetric] = {}
        self.labels: np.array = []
        self.user_data: UserData = None

    def get_metrics(self):
        return self.metrics

    def get_metric(self, name):
        return self.metrics[name]

    def get_labels(self):
        return self.labels

    def get_user_data(self) -> UserData:
        return self.user_data

    def set_labels(self, labels: np.array):
        self.labels = labels

    def set_metric(self, name: MetricNames, metric: InputMetric):
        self.metrics[name] = metric

    def set_user_data(self, user_data: UserData):
        self.user_data: UserData = user_data

    def get_metric_matrix(self):
        metrics = []
        names = []
        for name, metric in self.metrics.items():
            metrics.append(metric.get_value())
            names.append(name)
        # Returns shape of metrics (n_samples, n_features)
        return np.array(metrics).T, names
