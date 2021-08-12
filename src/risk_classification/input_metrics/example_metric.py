from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.EXAMPLE
METRIC_DATA_TYPE = MetricDataTypes.EXAMPLE


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        return sum(kwargs['data'])
