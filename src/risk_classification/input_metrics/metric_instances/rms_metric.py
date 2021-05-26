from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters


class Metric(RiskClassificationInputMetric):
    def generate_metric(self, **kwargs):
        motion_filters = MotionFilters()
        return motion_filters.calculate_rms(kwargs['data'])
