from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.GAIT_SPEED_ESTIMATOR


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super(RiskClassificationInputMetric, self).__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        # Initialize the gait speed metric
        gait_speed = None
        # Access data required for gait speed estimation from keyword arguments
        acc_data = kwargs['data']
        user_data = kwargs['user_data']
        ma_step_window_size = kwargs['ma_step_window_size']
        min_walk_dur = kwargs['min_walk_dur']
        # If the walking bout is long enough to detect step length
        if self.check_walking_duration(min_walk_dur):
            # Continue with gait estimation
            # Detect the heel strikes in the walking data
            strike_indexes = self.detect_heel_strikes(acc_data)
            # Estimate the stride length for each step
            step_lengths = self.estimate_stride_length(strike_indexes)
            # Estimate the gait speed from the stride lengths and timing between steps
            return self.estimate_gait_speed(step_lengths, user_data, ma_step_window_size)
        else:
            raise ValueError('Walking bout is not long enough to estimate gait speed')


    def detect_heel_strikes(self, data):
        strike_indexes = []
        return strike_indexes

    def estimate_stride_length(self, strike_indexes):
        # Heel strike index with the corresponding step length
        step_lengths = []
        return step_lengths

    def estimate_gait_speed(self, step_lengths, user_data, ma_step_window_size):
        # For given step lengths and the moving average step window size, calculate the
        # gait speed for this walking bout
        gait_speed = None
        return gait_speed

    def check_walking_duration(self, min_walk_dur):
        # Check how long the walking bout is, if it exceeds the minimum walking bout duration, then
        # it is valid, otherwise, it is invalid
        walk_bout_valid = False
        return walk_bout_valid

