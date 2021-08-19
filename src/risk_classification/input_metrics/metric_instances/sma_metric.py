from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.SIGNAL_MAGNITUDE_AREA
METRIC_DATA_TYPE = MetricDataTypes.VERTICAL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        return sum(kwargs['data'])