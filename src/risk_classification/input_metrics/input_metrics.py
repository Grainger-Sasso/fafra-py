from typing import Dict
from src.risk_classification.input_metrics.metric_names import MetricNames

from src.risk_classification.input_metrics.input_metric import InputMetric


class InputMetrics:
    def __init__(self):
        self.metrics: Dict[MetricNames: InputMetric] = []

    def get_metrics(self):
        return self.metrics

    def get_metric(self, name):
        return self.metrics[name]

    def set_metric(self, name: MetricNames, metric: InputMetric):
        self.metrics[name] = metric

    def get_metric_matrix(self):
        metrics = [metric for metric in self.metrics.values()]
        for metric in self.metrics.values():
            me
        pass
