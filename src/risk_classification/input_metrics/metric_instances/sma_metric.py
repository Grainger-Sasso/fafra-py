from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.EXAMPLE


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        #super(Metric, self).__init__(METRIC_NAME)
        #super(RiskClassificationInputMetric, self).__init__(METRIC_NAME)
        #super().__init__(METRIC_NAME)
        RiskClassificationInputMetric.__init__(self, METRIC_NAME)

    def generate_metric(self, **kwargs):
        return sum(kwargs['data'])