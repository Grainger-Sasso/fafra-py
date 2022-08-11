from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.gait_analysis.gait_analyzer_v2 import GaitAnalyzerV2


METRIC_NAME = MetricNames.GAIT_SPEED_ESTIMATOR
METRIC_DATA_TYPE = MetricDataTypes.USER_DATA


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        gait_speed, fig, gait_params = GaitAnalyzerV2().estimate_gait_speed(
            kwargs['data'], hpf=False, max_com_v_delta=0.14, plot_gait_cycles=False)
        return gait_speed
