from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.EXAMPLE


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        return kwargs
