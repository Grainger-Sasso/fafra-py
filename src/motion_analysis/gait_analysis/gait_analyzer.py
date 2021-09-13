import numpy as np

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.visualization_tools.gse_viz import GSEViz
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.motion_analysis.filters.motion_filters import MotionFilters


class GaitAnalyzer:

    def __init__(self):
        self.gse_viz = GSEViz()

    def estimate_gait_speed(self, user_data: UserData):
        """
        Several assumptions are made in this version of the gait speed
        estimator:
        1. The orientation of the sensor is such that the vertical axis of the
            accelerometer is parallel to the vertical axis of the body
        2. The heel strike is defined by the peak anteroposterior acceleration
            of the stride cycle (Ziljstra 2003)
        3. The vertical velocity upon each heel strike is approximately zero
        See Rispens et. al. 2021 for more information on the design of this
        gait speed estimator.
        https://www.mdpi.com/1424-8220/21/5/1854/htm#B13-sensors-21-01854
        :param user_data:
        :return:
        """
        # TODO: refactor this function and the user_data object to have
        #  segmented walking bouts, this function should assume to have
        #  walking bouts readily available
        # Access data required for gait speed estimation from keyword arguments
        lpf_v_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        # Given assumption 1, remove the effects of gravity from the vertical
        # acc data
        v_acc_data = lpf_v_data - 9.81
        ap_acc_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('anteroposterior')
        user_height = user_data.get_clinical_demo_data().get_height()
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        # Detect the peaks (heel strikes) in the walking data
        v_peak_indexes = self._detect_peaks(v_acc_data)
        ap_peak_indexes = self._detect_peaks(ap_acc_data)
        step_lengths, v_displacement = self._estimate_step_lengths(
            v_acc_data, samp_freq, ap_peak_indexes, user_height)
        total_distance = step_lengths.sum()
        total_time = len(v_displacement)/samp_freq
        gait_speed = (total_distance/total_time)
        # self.gse_viz.plot_gse_results(user_data, v_peak_indexes,
        #                               ap_peak_indexes, v_displacement)
        return gait_speed

    def _detect_peaks(self, acc_data):
        strike_indexes = PeakDetector().detect_peaks(acc_data)
        return strike_indexes

    def _estimate_step_lengths(self, v_acc, samp_freq,
                              strike_indexes, user_height):
        # See Frisancho et al. 2007 for leg length estimation
        # https://journals.sagepub.com/doi/pdf/10.1177/1545968314532031
        leg_length = 0.48 * user_height
        # Initialize list of step lengths in walking bout
        step_lengths = []
        # Initialize vertical displacement for whole walking bout
        v_displacement = []
        # For every step (interval between ap peak)
        step_start_ixs = strike_indexes[:-1]
        step_end_ixs = strike_indexes[1:]
        for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
            # Calculate the vertical displacement of that step
            step_v_disp = self._estimate_step_v_displacement(v_acc, start_ix,
                                                  end_ix, samp_freq)
            # Add step vertical displacement to the walking bout
            # vertical displacment
            v_displacement.extend(step_v_disp)
            # Compute the difference between the largest and smallest vertical
            # displacement of CoM
            h = max(step_v_disp) - min(step_v_disp)
            # Formula for step length derived from inverted pendulum model
            step_length = 1.25 * 2 * (((2 * leg_length - h) * h) ** 0.5)
            # Apply correction for mediolateral component of step length
            step_length = ((step_length ** 2) - ((0.094*leg_length) ** 2)) ** 0.5
            step_lengths.append(step_length)
        return np.array(step_lengths), v_displacement

    def _estimate_step_v_displacement(self, v_acc, start_ix,
                                      end_ix, samp_freq):
        period = 1 / samp_freq
        # The initial position of the CoM at t=0 is arbitrary, set to 0
        p0 = 0.0
        # Given assumption 3 of estimate_gait_speed(), we assume initial
        # velocity at each heel strike to be zero
        v0 = 0.0
        acc = v_acc[start_ix:(end_ix - 1)]
        vel = self._compute_single_integration(acc, period, v0)
        vel = MotionFilters().apply_lpass_filter(vel, 0.5,
                                                 samp_freq, 'high')
        pos = self._compute_single_integration(vel[:-1], period, p0)
        pos = MotionFilters().apply_lpass_filter(pos, 0.5,
                                                 samp_freq, 'high')
        return pos

    def _compute_single_integration(self, data, period, x0):
        # single integration for time series data is the sum of (the
        # product of the signal at time t and the sample period) and
        # (the current integrated value at time t)
        x = [x0]
        for i in data:
            x_t = i * period + x0
            x.append(x_t)
            x0 = x_t
        return x

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


