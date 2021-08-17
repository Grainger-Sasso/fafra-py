from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.feature_extraction.gait.gait_analyzer import GaitAnalyzer


METRIC_NAME = MetricNames.GAIT_SPEED_ESTIMATOR
METRIC_DATA_TYPE = MetricDataTypes.ALL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        return GaitAnalyzer().estimate_gait_speed(kwargs['data'])
