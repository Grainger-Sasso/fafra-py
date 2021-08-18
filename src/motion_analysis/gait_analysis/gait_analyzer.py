from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.visualization_tools.gse_viz import GSEViz
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class GaitAnalyzer:

    def __init__(self):
        self.ma_step_window_size = 5
        self.min_walk_dur = 3.0
        self.gse_viz = GSEViz()

    def estimate_gait_speed(self, user_data: UserData):
        # Initialize the gait speed metric
        gait_speed = None
        # Access data required for gait speed estimation from keyword arguments
        v_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        ml_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('mediolateral')
        ap_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('anteroposterior')
        user_height = user_data.get_clinical_demo_data().get_height()
        self.gse_viz.plot_gse_results(user_data, [1])
        # user_data = kwargs['user_data']
        # If the walking bout is long enough to detect step length
        if self.check_walking_duration():
            # Continue with gait estimation
            # Detect the heel strikes in the walking data
            strike_indexes = self.detect_heel_strikes(v_acc_data, ml_acc_data, ap_acc_data)
            # Estimate the stride length for each step
            step_lengths = self.estimate_stride_length(strike_indexes)
            # Estimate the gait speed from the stride lengths and timing between steps
            return self.estimate_gait_speed(step_lengths, user_height)
        else:
            raise ValueError('Walking bout is not long enough to estimate gait speed')

        # For given step lengths and the moving average step window size, calculate the
        # gait speed for this walking bout
        print(self.ma_step_window_size)
        gait_speed = None
        return gait_speed

    def detect_heel_strikes(self, v_acc_data, ml_acc_data, ap_acc_data):
        strike_indexes = []
        return strike_indexes

    def estimate_stride_length(self, strike_indexes):
        # Heel strike index with the corresponding step length
        step_lengths = []
        return step_lengths

    def check_walking_duration(self):
        # Check how long the walking bout is, if it exceeds the minimum walking bout duration, then
        # it is valid, otherwise, it is invalid
        print(self.ma_step_window_size)
        print(self.min_walk_dur)
        walk_bout_valid = False
        return walk_bout_valid

    def detect_gait(self, data):
        """
        http://www.l3s.de/~anand/tir14/lectures/ws14-tir-foundations-2.pdf
        :param data:
        :return:
        """
        # Run kalman filter on the data
        # Take unbiased vertical acceleration
        # Perform discrete fourier transform to detect possible periodic data
        # Run auto-correlation to remove false-positives
        pass

