import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.datasets.ltmm.ltmm_dataset import LTMMData


METRIC_NAME = MetricNames.GAIT_SPEED_ESTIMATOR
METRIC_DATA_TYPE = MetricDataTypes.LTMM


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)
        self.ma_step_window_size = 5
        self.min_walk_dur = 3.0


    def generate_metric(self, **kwargs):
        # Initialize the gait speed metric
        gait_speed = None
        # Access data required for gait speed estimation from keyword arguments
        ltmm_data: LTMMData = kwargs['data']
        v_acc_data = ltmm_data.get_axis_acc_data('vertical')
        ml_acc_data = ltmm_data.get_axis_acc_data('mediolateral')
        ap_acc_data = ltmm_data.get_axis_acc_data('anteroposterior')
        user_height = ltmm_data.get_height()
        # user_data = kwargs['user_data']
        # If the walking bout is long enough to detect step length
        if self.check_walking_duration():
            # Continue with gait estimation
            # Detect the heel strikes in the walking data
            strike_indexes = self.detect_heel_strikes(acc_data)
            # Estimate the stride length for each step
            step_lengths = self.estimate_stride_length(strike_indexes)
            # Estimate the gait speed from the stride lengths and timing between steps
            return self.estimate_gait_speed(step_lengths, user_height)
        else:
            raise ValueError('Walking bout is not long enough to estimate gait speed')


    def detect_heel_strikes(self, data):
        strike_indexes = []
        return strike_indexes

    def estimate_stride_length(self, strike_indexes):
        # Heel strike index with the corresponding step length
        step_lengths = []
        return step_lengths

    def estimate_gait_speed(self, step_lengths, user_height):
        # For given step lengths and the moving average step window size, calculate the
        # gait speed for this walking bout
        print(self.ma_step_window_size)
        gait_speed = None
        return gait_speed

    def check_walking_duration(self):
        # Check how long the walking bout is, if it exceeds the minimum walking bout duration, then
        # it is valid, otherwise, it is invalid
        print(self.ma_step_window_size)
        print(self.min_walk_dur)
        walk_bout_valid = False
        return walk_bout_valid
