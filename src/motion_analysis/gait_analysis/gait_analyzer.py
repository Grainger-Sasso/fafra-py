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
        """
        Several assumptions are made in this version of the gait speed
        estimator:
        1. The orientation of the sensor is such that the vertical axis of the
            accelerometer is parallel to the vertical axis of the body
        2. The heel strike is defined by the peak anteroposterior acceleration
            of the stride cycle
        3. The vertical velocity upon each heel strike is approximately zero
        :param user_data:
        :return:
        """
        # TODO: refactor this funciton adn the user_data object to have
        #  segmented walking bouts, this funciton should assume to have
        #  walking bouts readily availble
        # Initialize the gait speed metric
        gait_speed = None
        # Access data required for gait speed estimation from keyword arguments
        raw_v_data = user_data.get_imu_data()[IMUDataFilterType.RAW].get_acc_axis_data('vertical')
        v_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        # Given assumption 1, remove the effects of gravity from the vertical
        # acc data
        v_acc_data = v_acc_data - 9.81
        ml_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('mediolateral')
        ap_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('anteroposterior')
        user_height = user_data.get_clinical_demo_data().get_height()
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        # user_data = kwargs['user_data']
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

    def detect_peaks(self, acc_data):
        strike_indexes = PeakDetector().detect_peaks(acc_data)
        return strike_indexes

    def compute_v_displacement(self, v_acc, samp_freq, ap_peak_ix):
        # Define params for double integration
        period = 1/samp_freq
        # Initialize variable for the whole walking bout vertical displacement
        whole_disp = []
        # For every step (interval between ap peak)
        step_start_ixs = ap_peak_ix[:-1]
        step_end_ixs = ap_peak_ix[1:]
        for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
            # The initial position of the CoM at t=0 is arbitrary, set to 0
            p0 = 0.0
            # Given assumption 3 of estimate_gait_speed(), we assume initial
            # velocity at each heel strike to be zero
            v0 = 0.0
            acc = v_acc[start_ix:(end_ix - 1)]
            vel = self.compute_single_integration(acc, period, v0)
            vel = MotionFilters().apply_lpass_filter(vel, 0.5,
                                               samp_freq, 'high')
            vel = vel[:-1]
            pos = self.compute_single_integration(vel, period, p0)
            pos = MotionFilters().apply_lpass_filter(pos, 0.5,
                                                     samp_freq, 'high')
            whole_disp.extend(pos)
        print(len(v_acc))
        print(len(whole_disp))
        return whole_disp

    def compute_single_integration(self, data, period, x0):
        # single integration for time series data is the sum of (the
        # product of the signal at time t and the sample period) and
        # (the current integrated value at time t)
        x = [x0]
        for i in data:
            x_t = i * period + x0
            x.append(x_t)
            x0 = x_t
        return x

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


