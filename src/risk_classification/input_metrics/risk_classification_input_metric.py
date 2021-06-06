from abc import ABC, abstractmethod

from src.risk_classification.input_metrics.metric_names import MetricNames


class RiskClassificationInputMetric(ABC):
    def __init__(self, metric_name: MetricNames):
        self.metric_name = metric_name

    def get_metric_name(self):
        return self.metric_name

    @abstractmethod
    def generate_metric(self, **kwargs):
        pass
