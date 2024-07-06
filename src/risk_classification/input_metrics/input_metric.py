import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames


class InputMetric:
    def __init__(self, name: MetricNames, value: np.array):
        self.name = name
        self.value: np.array = value

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value
