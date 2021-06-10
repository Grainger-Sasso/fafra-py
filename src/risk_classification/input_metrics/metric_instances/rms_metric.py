from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters


METRIC_NAME = MetricNames.ROOT_MEAN_SQUARE


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        motion_filters = MotionFilters()
        return motion_filters.calculate_rms(kwargs['data'])
