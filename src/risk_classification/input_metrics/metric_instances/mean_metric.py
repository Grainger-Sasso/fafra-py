import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.MEAN


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME)
        self.data_type = 'vertical'

    def get_data_type(self):
        return self.data_type

    def generate_metric(self, **kwargs):
        return np.mean(kwargs['data'])
