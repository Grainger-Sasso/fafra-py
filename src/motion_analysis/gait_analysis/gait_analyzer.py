import numpy as np
import math
from matplotlib import pyplot as plt

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
            accelerometer is parallel to the vertical axis of the body, i.e. we
            do not need to transform the accelerometer axes from sensor to
            global
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
        lpf_v_data = user_data.get_imu_data(IMUDataFilterType.LPF).get_acc_axis_data('vertical')
        # Given assumption 1, remove the effects of gravity from the vertical
        # acc data
        v_acc_data = lpf_v_data - 9.80665
        ap_acc_data = user_data.get_imu_data(IMUDataFilterType.LPF).get_acc_axis_data('anteroposterior')
        user_height = user_data.get_clinical_demo_data().get_height()
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        # Detect the peaks (heel strikes) in the walking data
        v_peak_indexes = self._detect_peaks(v_acc_data)
        ap_peak_indexes = self._detect_peaks(ap_acc_data)
        step_lengths, v_displacement, valid_strike_ixs, invalid_strike_ixs, tot_time = self._estimate_step_lengths(
            v_acc_data, samp_freq, ap_peak_indexes, user_height)
        total_distance = step_lengths.sum()
        gait_speed = (total_distance/tot_time)
        self.plot_gait_cycles(v_displacement, valid_strike_ixs, invalid_strike_ixs, samp_freq)
        # self.gse_viz.plot_gse_results(user_data, v_peak_indexes,
        #                               ap_peak_indexes, v_displacement)
        return gait_speed

    def plot_gait_cycles(self, v_disp, valid_ix, invalid_ix, samp_freq):
        # Create time axis
        time = np.linspace(0.0, len(v_disp)/samp_freq, len(v_disp))
        plt.plot(time, v_disp)
        plt.plot(np.array(time)[valid_ix].tolist(), np.array(v_disp)[valid_ix].tolist(), 'b^')
        plt.plot(np.array(time)[invalid_ix].tolist(), np.array(v_disp)[invalid_ix].tolist(), 'rv')
        plt.show()
        print('a')

    def _detect_peaks(self, acc_data):
        strike_indexes = PeakDetector().detect_peaks(acc_data)
        return strike_indexes

    def _estimate_step_lengths(self, v_acc, samp_freq,
                              strike_indexes, user_height):
        # See Frisancho et al. 2007 for leg length estimation
        # https://journals.sagepub.com/doi/pdf/10.1177/1545968314532031
        valid_strike_ixs = []
        invalid_strike_ixs = []
        tot_time = 0.0
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
            step_v_disp = self.estimate_v_displacement(v_acc, start_ix,
                                                  end_ix, samp_freq)
            # Add step vertical displacement to the walking bout
            # vertical displacment
            v_displacement.extend(step_v_disp)
            # Compute the difference between the largest and smallest vertical
            # displacement of CoM
            h = max(step_v_disp) - min(step_v_disp)
            # Introduce a check to make sure that COM displacement is less than
            # an acceptable max value: https://journals.physiology.org/doi/full/10.1152/japplphysiol.00103.2005#:~:text=The%20average%20vertical%20displacement%20of,speeds%20(P%20%3D%200.0001).
            # (filters out erroneous COM displacement values)
            if h < 0.3:
                # Formula for step length derived from inverted pendulum model
                step_lengths.append(self._calc_step_length(h, leg_length))
                # Consider step indices valid
                valid_strike_ixs.append(len(v_displacement)-1)
                # Increment the total time up
                tot_time += ((end_ix - start_ix) * samp_freq)
            else:
                # Consider the step indices invalid
                invalid_strike_ixs.append(len(v_displacement)-1)
        # Remove duplicates from strike indices
        valid_strike_ixs = list(set(valid_strike_ixs))
        invalid_strike_ixs = list(set(invalid_strike_ixs))
        if np.isnan(step_lengths).any():
            print('AAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHH')
        return np.array(step_lengths), v_displacement, valid_strike_ixs, invalid_strike_ixs, tot_time

    def _calc_step_length(self, h, leg_length):
        g = ((2 * leg_length - h) * h)
        step_length = 1.25 * 2 * (g ** 0.5)
        # Apply correction for mediolateral component of step length
        if ((step_length ** 2) > ((0.094 * leg_length) ** 2)):
            step_length = ((step_length ** 2) - ((0.094 * leg_length) ** 2)) ** 0.5
        return step_length


    def estimate_v_displacement(self, v_acc, start_ix,
                                      end_ix, samp_freq):
        period = 1 / samp_freq
        # The initial position of the CoM at t=0 is arbitrary, set to 0
        p0 = 0.0
        # Given assumption 3 of estimate_gait_speed(), we assume initial
        # velocity at each heel strike to be zero
        v0 = 0.0
        acc = v_acc[start_ix:(end_ix - 1)]
        vel = self._compute_single_integration(acc, period, v0)
        # TODO: investigate a more appropriate cut-off frequency for the high-pass filter, atm the filter is confounding the results/is not usefult for preventing integration drift
        # vel = MotionFilters().apply_lpass_filter(vel, 0.5,
        #                                          samp_freq, 'high')
        pos = self._compute_single_integration(vel[:-1], period, p0)
        # pos = MotionFilters().apply_lpass_filter(pos, 0.5,
        #                                          samp_freq, 'high')
        return pos

    def _compute_single_integration(self, data, period, x0):
        # single integration for time series data is the sum of (the
        # product of the signal at time t and the sample period) and
        # (the current integrated value at time t)
        x = [x0]
        x_i = x0
        for i in data:
            x_t = i * period + x_i
            x.append(x_t)
            x_i = x_t
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


