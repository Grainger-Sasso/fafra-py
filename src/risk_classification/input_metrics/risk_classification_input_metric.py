from abc import ABC, abstractmethod

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes



class RiskClassificationInputMetric(ABC):
    def __init__(self, metric_name: MetricNames, data_type: MetricDataTypes):
        self.metric_name = metric_name
        self.data_type = data_type

    def get_metric_name(self):
        return self.metric_name

    def get_data_type(self):
        return self.data_type

    @abstractmethod
    def generate_metric(self, **kwargs):
        pass
