import numpy as np

from src.risk_classification.input_metrics.risk_classification_input_metrics import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.datasets.ltmm.ltmm_dataset import LTMMData


class RMSMetric(RiskClassificationInputMetric):
    def __init__(self):
        self.motion_filters = MotionFilters()

    def generate_metric(self, ltmm_data: LTMMData):
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        return self.motion_filters.calculate_rms(v_axis_data)
