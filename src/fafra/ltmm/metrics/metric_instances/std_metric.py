import numpy as np

from src.risk_classification.input_metrics.risk_classification_input_metrics import RiskClassificationInputMetric
from src.datasets.ltmm.ltmm_dataset import LTMMData


class STDMetric(RiskClassificationInputMetric):
    def generate_metric(self, ltmm_data: LTMMData):
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        return np.std(v_axis_data)
