import numpy as np

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
            v_displacement = self.compute_v_displacement(v_acc_data, samp_freq,
                                                       ap_peak_indexes)
            step_lengths = self.estimate_step_lengths(ap_peak_indexes,
                                                          v_displacement)
            total_distance = step_lengths.sum()
            total_time = len(v_displacement)/samp_freq
            gait_speed = (total_distance/total_time)*1.25
            self.gse_viz.plot_gse_results(user_data, v_peak_indexes,
                                          ap_peak_indexes, v_displacement)
            print(gait_speed)
            return gait_speed
        else:
            raise ValueError('Walking bout is not long enough to estimate gait speed')

        # For given step lengths and the moving average step window size, calculate the
        # gait speed for this walking bout
        gait_speed = None
        return gait_speed

    def detect_peaks(self, acc_data):
        strike_indexes = PeakDetector().detect_peaks(acc_data)
        return strike_indexes

    def compute_v_displacement(self, v_acc, samp_freq, ap_peak_ix):
        # Convert the vertical acceleration from g to m/s^2
        # v_acc = v_acc * 9.81
        # Define params for double integration
        period = 1/samp_freq
        # Initialize variable for the whole walking bout vertical displacement
        whole_disp = []
        # For every step (interval between ap peak)
        step_start_ixs = ap_peak_ix[:-1]
        step_end_ixs = ap_peak_ix[1:]
        for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
            p0 = 0.0
            v0 = 0.0
            displacement = [0.0]
            for acc in v_acc[start_ix:(end_ix-1)]:
                v_t = acc*period + v0
                p_t = (acc/2)*(period**2) + (v0*period) + p0
                displacement.append(p_t)
                p0 = p_t
                v0 = v_t
            whole_disp.extend(displacement)
        whole_disp = MotionFilters().apply_lpass_filter(whole_disp, 0.5,
                                                          samp_freq, 'high')
        return whole_disp

    def estimate_step_lengths(self, strike_indexes, displacement,
                              leg_length=0.75):
        # Todo: fix the leg length estimation
        # Heel strike index with the corresponding step length
        step_lengths = []
        step_start_ixs = strike_indexes[:-1]
        step_end_ixs = strike_indexes[1:]
        for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
            # parameter, h, defined as the difference between the largest and
            # smallest vertical position of CoM
            h = max(displacement[start_ix:end_ix]) - min(displacement[start_ix:end_ix])
            # Formula for step length derived from inverted pendulum model
            step_length = 2 * (((2 * leg_length * h) - (h ** 2)) ** 0.5)
            step_lengths.append(step_length)
        return np.array(step_lengths)

    def check_walking_duration(self):
        # Check how long the walking bout is, if it exceeds the minimum walking bout duration, then
        # it is valid, otherwise, it is invalid
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


