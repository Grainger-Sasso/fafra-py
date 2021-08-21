from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.visualization_tools.gse_viz import GSEViz
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.motion_analysis.filters.motion_filters import MotionFilters


class GaitAnalyzer:

    def __init__(self):
        self.ma_step_window_size = 5
        self.min_walk_dur = 3.0
        self.gse_viz = GSEViz()

    def estimate_gait_speed(self, user_data: UserData):
        # Initialize the gait speed metric
        gait_speed = None
        # Access data required for gait speed estimation from keyword arguments
        raw_v_data = user_data.get_imu_data()[IMUDataFilterType.RAW].get_acc_axis_data('vertical')
        v_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        ml_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('mediolateral')
        ap_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('anteroposterior')
        user_height = user_data.get_clinical_demo_data().get_height()
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        # user_data = kwargs['user_data']
        # If the walking bout is long enough to detect step length
        if self.check_walking_duration():
            # Continue with gait estimation
            # Detect the peaks (heel strikes) in the walking data
            v_peak_indexes = self.detect_peaks(v_acc_data)
            ap_peak_indexes = self.detect_peaks(ap_acc_data)
            displacement = self.compute_v_displacement(v_acc_data,
                                                       user_height, samp_freq)
            self.gse_viz.plot_gse_results(user_data, v_peak_indexes,
                                          ap_peak_indexes, displacement)
            # # Estimate the stride length for each step
            # step_lengths = self.estimate_stride_length(strike_indexes)
            # # Estimate the gait speed from the stride lengths and timing between steps
            # return self.estimate_gait_speed(step_lengths, user_height)
            return 1.0
        else:
            raise ValueError('Walking bout is not long enough to estimate gait speed')

        # For given step lengths and the moving average step window size, calculate the
        # gait speed for this walking bout
        print(self.ma_step_window_size)
        gait_speed = None
        return gait_speed

    def detect_peaks(self, acc_data):
        strike_indexes = PeakDetector().detect_peaks(acc_data)
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
        walk_bout_valid = True
        return walk_bout_valid

    def detect_gait(self, data):
        """
        http://www.l3s.de/~anand/tir14/lectures/ws14-tir-foundations-2.pdf
        :param data:
        :return:
        """
        # Run kamlan filter on the data
        # Take unbiased vertical acceleration
        # Perform discrete fourier transform to detect possible periodic data
        # Run auto-correlation to remove false-positives
        pass

    def compute_v_displacement(self, v_acc, height, samp_freq):
        displacement = []
        p0 = height/2
        v0 = 0.0
        for acc in v_acc:
            v_t = acc*samp_freq* + v0
            p_t = (acc/2)*(samp_freq**2) + (v0*samp_freq) + p0
            displacement.append(p_t)
            p0 = p_t
            v0 = v_t
        displacement = MotionFilters().apply_lpass_filter(displacement, 0.1, samp_freq, 'high')
        return displacement

