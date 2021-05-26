import numpy as np

from src.risk_classification.input_metrics.risk_classification_input_metrics import RiskClassificationInputMetric


class Metric(RiskClassificationInputMetric):
    def generate_metric(self, **kwargs):
        return np.std(kwargs['data'])
