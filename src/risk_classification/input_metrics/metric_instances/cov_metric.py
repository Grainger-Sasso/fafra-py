import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.EXAMPLE


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        RiskClassificationInputMetric.__init__(self, METRIC_NAME)

    def generate_metric(self, **kwargs):
        return np.std(kwargs['data']) / np.mean(kwargs['data'])
