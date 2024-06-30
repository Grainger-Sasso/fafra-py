import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.visualization_tools.gse_viz import GSEViz
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.input_metrics.metric_instances.fft_peak_metric import Metric


class GaitAnalyzerV2:

    def __init__(self):
        self.gse_viz = GSEViz()
        self.valid_strike_ixs = []
        self.invalid_strike_ixs = []

    def estimate_gait_speed(self, user_data: UserData, hpf, max_com_v_delta, plot_gait_cycles=False):
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

        See Ziljstra et. al. 2003 and Rispens et. al. 2021 for more information
        on the design of this gait speed estimator.
        https://pure.rug.nl/ws/portalfiles/portal/6673749/Zijlstra_2003_Gait_Posture.pdf
        https://www.mdpi.com/1424-8220/21/5/1854/htm#B13-sensors-21-01854
        :param user_data:
        :return:
        """
        # Access data required for gait speed estimation
        lpf_data = user_data.get_imu_data(IMUDataFilterType.LPF)
        ml_acc_data = lpf_data.get_acc_axis_data('mediolateral')
        ap_acc_data = lpf_data.get_acc_axis_data('anteroposterior')
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        # Apply FFT to ap signal to evaluate periodicity of data
        ap_signal_fft = Metric().generate_metric(data=ap_acc_data, sampling_frequency=samp_freq)
        # Filter out any walking bouts with cadence less than 0.115 Hz
        if ap_signal_fft > 0.115:
            # See Frisancho et al. 2007 for leg length estimation
            # https://journals.sagepub.com/doi/pdf/10.1177/1545968314532031
            user_height = user_data.get_clinical_demo_data().get_height()
            leg_length = 0.48 * user_height
            # Access the vertical acceleration data and detrend the data
            v_acc_data = lpf_data.get_acc_axis_data('vertical')
            v_acc_data = v_acc_data - np.mean(v_acc_data)
            # Detect the heel strikes in the walking data from peaks in ap axis
            heel_strike_indexes = self._detect_peaks(ap_acc_data)
            # Validate that the peaks detected are valid based on criteria for
            # spacing of steps
            valid_strike_ix_clusters = self._validate_strike_ixs(
                heel_strike_indexes, ap_acc_data, ap_signal_fft,
                samp_freq, v_acc_data)
            valid_strike_ixs = sorted(list(
                set([cluster for clusters in valid_strike_ix_clusters for cluster
                     in clusters])))
            # Given assumption 1, remove the effects of gravity from the
            # vertical acc data
            # Calculate the vertical displacement of the COM
            if valid_strike_ixs:

                diffs = [a - b for a, b in
                         zip(valid_strike_ixs[1:], valid_strike_ixs[:-1])]
                whole_v_disp = self.estimate_all_v_displacement(
                    valid_strike_ix_clusters, v_acc_data, samp_freq, hpf)
                gait_speed, all_com_v_deltas, invalid_strike_indexes, all_step_lengths = self.estimate_all_step_lengths(
                    valid_strike_ix_clusters, whole_v_disp, samp_freq, leg_length, max_com_v_delta)
                additional_invalid_strike_ixs = [ix for ix in heel_strike_indexes if ix not in valid_strike_ixs]
                invalid_strike_ixs = list(set(invalid_strike_indexes + additional_invalid_strike_ixs))
                if plot_gait_cycles:
                    fig = self.plot_gait_cycles(
                        v_acc_data, ml_acc_data, ap_acc_data, whole_v_disp, invalid_strike_indexes, samp_freq,
                        all_com_v_deltas, valid_strike_ix_clusters)
                else:
                    fig = None
                gait_params = {'cadence': valid_strike_ix_clusters,
                               'step_lengths': all_step_lengths,
                               'v_com_disps': all_com_v_deltas}
            else:
                gait_speed = np.nan
                fig = None
                gait_params = None
        else:
            gait_speed = np.nan
            fig = None
            gait_params = None
        return gait_speed, fig, gait_params

    def _detect_peaks(self, acc_data):
        height = None
        threshold = None
        distance = None
        prominence = 1.2
        width = None
        wlen = None
        rel_height = 0.5
        plateau_size = None
        peaks = PeakDetector().detect_peaks(acc_data, height=height,
                threshold=threshold, distance=distance,prominence=prominence,
                width=width, wlen=wlen, rel_height=rel_height,
                                            plateau_size=plateau_size)
        return peaks[0]

    def _validate_strike_ixs(self, heel_strike_ixs, ap_acc_data, ap_fft_peak,
                             samp_freq, v_acc_data, min_cluster_size=6):
        # Initialize variables
        valid_strike_ixs = []
        valid_step_interval = 1 / ap_fft_peak
        step_interval_tolerance = valid_step_interval * 0.90
        min_step_interval = valid_step_interval - step_interval_tolerance
        max_step_interval = valid_step_interval + step_interval_tolerance
        step_start_ixs = heel_strike_ixs[:-1]
        step_end_ixs = heel_strike_ixs[1:]
        # Check that the peaks detected fall in interval provided by FFT
        for start_ix, stop_ix in zip(step_start_ixs, step_end_ixs):
            step_interval = (stop_ix - start_ix) / samp_freq
            if min_step_interval < step_interval < max_step_interval:
                valid_strike_ixs.append(start_ix)
        # Check the fit of vertical acceleration values between each step
        # indexes to objective function and remove indexes that have poor fit

        # Check that the steps are consecutive as defined by the number of times
        # the gradient changes sign between strike ixs
        end_cluster_ixs = []
        count = 0
        for start_ix, stop_ix in zip(valid_strike_ixs[:-1], valid_strike_ixs[1:]):
            # Compute the gradient of the AP signal between start and stop ix
            ap_data = ap_acc_data[start_ix:stop_ix]
            ap_data_gradient = np.gradient(ap_data)
            # Count zero crossings between them
            ap_zero_crossings = np.where(np.diff(np.sign(ap_data_gradient)))[0]
            ap_num_zero_crossings = len(ap_zero_crossings)
            # Compute the gradient of the V signal between start and stop ix
            v_data = v_acc_data[start_ix:stop_ix]
            v_data_gradient = np.gradient(v_data)
            # Count zero crossings between them
            v_zero_crossings = np.where(np.diff(np.sign(v_data_gradient)))[0]
            v_num_zero_crossings = len(v_zero_crossings)
            # If count is 3 or more
            if ap_num_zero_crossings >= 3:
                # Add the count to the end cluster ixs
                end_cluster_ixs.append(count)
            elif v_num_zero_crossings >= 3:
                end_cluster_ixs.append(count)
            # Increment count
            count += 1
        # Initialize stepping clusters variable
        step_clusters = []
        step_cluster = []
        # Clusters are defined as all valid strike ixs where the end index of
        # the indexes defines the end of the cluster
        for idx, value in enumerate(valid_strike_ixs):
            step_cluster.append(value)
            if idx in end_cluster_ixs:
                step_clusters.append(step_cluster)
                step_cluster = []
        step_clusters.append(step_cluster)
        step_clusters = [cluster for cluster in
                         step_clusters if len(cluster) > min_cluster_size]
        # Check that the clusters contain more the n steps
        valid_step_clusters = [cluster for cluster in step_clusters if
                               len(cluster) >= 10]
        return valid_step_clusters

    def estimate_all_v_displacement(self, heel_strike_ix_clusters, v_acc, samp_freq, hpf):
        whole_v_disp = []
        no_v_disp_ix = 0
        for cluster in heel_strike_ix_clusters:
            step_start_ixs = cluster[:-1]
            step_end_ixs = cluster[1:]
            v_displacement = []
            # For every step (interval between ap peak)
            for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
                # Calculate the vertical displacement of that step
                step_v_disp = self.estimate_v_displacement(v_acc, start_ix,
                                                           end_ix, samp_freq,
                                                           hpf)
                v_disp_mean = np.array(step_v_disp).mean()
                # Smooth the vertical displacement signal
                step_v_disp = self.smooth_step_v_disp(step_v_disp, start_ix, end_ix)
                new_v_disp_mean = np.array(step_v_disp).mean()
                step_v_disp = step_v_disp * (v_disp_mean / new_v_disp_mean)
                # Add step vertical displacement to the walking bout
                # vertical displacment
                v_displacement.extend(step_v_disp)
            no_v_disp_data = np.zeros((step_start_ixs[0] - no_v_disp_ix))
            whole_v_disp.extend(no_v_disp_data)
            whole_v_disp.extend(v_displacement)
            no_v_disp_ix = step_end_ixs[-1]
        no_v_disp_data = np.zeros((len(v_acc) - no_v_disp_ix))
        whole_v_disp.extend(no_v_disp_data)
        return whole_v_disp

    def smooth_step_v_disp(self, step_v_disp, start_ix, end_ix):
        filter_size = 21
        polyorder = 3
        if filter_size >= (end_ix - start_ix):
            filter_size = (end_ix - start_ix) - 1
        if (filter_size % 2) == 0:
            filter_size -= 1
        if polyorder < filter_size:
            step_v_disp = savgol_filter(step_v_disp, filter_size, polyorder)
        else:
            step_v_disp = np.array(step_v_disp)
        return step_v_disp

    def estimate_v_displacement(self, v_acc, start_ix,
                                      end_ix, samp_freq, hpf):
        period = 1 / samp_freq
        # The initial position of the CoM at t=0 is arbitrary, set to 0
        p0 = 0.0
        # Given assumption 3 of estimate_gait_speed(), we assume initial
        # vertical velocity at each heel strike to be zero
        v0 = 0.0
        acc = v_acc[start_ix:(end_ix - 1)]
        vel = self._compute_single_integration(acc, period, v0)
        # TODO: investigate a more appropriate cut-off frequency for the high-pass filter, atm the filter is confounding the results/is not usefult for preventing integration drift
        if hpf:
            vel = MotionFilters().apply_lpass_filter(vel, 0.1,
                                                     samp_freq, 'highpass')
        pos = self._compute_single_integration(vel[:-1], period, p0)
        if hpf:
            pos = MotionFilters().apply_lpass_filter(pos, 0.1,
                                                     samp_freq, 'highpass')
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

    def estimate_all_step_lengths(self, heel_strike_ix_clusters, whole_v_disp, samp_freq, leg_length, max_com_v_delta):
        # Initialize the gait speed estimation variable
        cluster_gait_speeds = []
        all_com_v_deltas = []
        invalid_step_indexes = []
        all_step_lengths = []
        for cluster in heel_strike_ix_clusters:
            step_start_ixs = cluster[:-1]
            step_end_ixs = cluster[1:]
            step_lengths, tot_time, com_v_deltas, invalid_strike_ixs = self.estimate_step_lengths(
                whole_v_disp, samp_freq, step_start_ixs, step_end_ixs,
                leg_length, max_com_v_delta)
            invalid_step_indexes.extend(invalid_strike_ixs)
            all_step_lengths.extend(step_lengths)
            distance = step_lengths.sum()
            gs = distance / tot_time
            cluster_gait_speeds.append(gs)
            all_com_v_deltas.extend(com_v_deltas)
        invalid_step_indexes = list(set(invalid_step_indexes))
        # Averages gait speed for all walking clusters, returns nan if
        # no step clusters were found
        gait_speed = np.array(cluster_gait_speeds).mean()
        return gait_speed, all_com_v_deltas, invalid_step_indexes, all_step_lengths

    def estimate_step_lengths(self, v_displacement, samp_freq,
                               step_start_ixs, step_end_ixs, leg_length,
                               max_com_v_delta):
        # Initialize plotting variables
        valid_strike_ixs = []
        invalid_strike_ixs = []
        com_v_deltas = []
        # Initialize step lengths in walking bout and total time spent walking
        tot_time = 0.0
        step_lengths = []
        # For every step (interval between ap peak)
        # Trying new method to calculate v_disps after smoothing the v disp data
        for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
            step_v_disp = v_displacement[start_ix: end_ix]
            # Compute the difference between the largest and smallest vertical
            # displacement of CoM
            com_v_delta = max(step_v_disp) - min(step_v_disp)
            com_v_deltas.append(com_v_delta)
            # Introduce a check to make sure that COM displacement is less than
            # an acceptable max value (0.08m = 8cm):
            # https://journals.physiology.org/doi/full/10.1152/japplphysiol.00103.2005#:~:text=The%20average%20vertical%20displacement%20of,speeds%20(P%20%3D%200.0001).
            # (filters out erroneous COM displacement values)
            # Formula for step length derived from inverted pendulum model
            if com_v_delta < max_com_v_delta:
                step_lengths.append(
                    self._calc_step_length(com_v_delta, leg_length))
                # Increment the total time up
                tot_time += ((end_ix - start_ix) / samp_freq)
            else:
                invalid_strike_ixs.append(start_ix)
        com_v_deltas = np.array(com_v_deltas)
        self._check_step_lengths(step_lengths)
        return np.array(step_lengths), tot_time, com_v_deltas.tolist(), invalid_strike_ixs

    def _calc_step_length(self, v_disp, leg_length):
        # Estimate of step length same as Ziljstra 2003, no empirical correction factor
        g = ((2 * leg_length - v_disp) * v_disp)
        step_length = 2 * (g ** 0.5)
        return step_length

    def _check_step_lengths(self, step_lengths):
        if np.isnan(step_lengths).any():
            raise ValueError(f'Computed step lengths contain erroneous value {str(step_lengths)}')

    def plot_gait_cycles(self, v_acc, ml_acc, ap_acc, v_disp, invalid_ix,
                         samp_freq, com_v_deltas, heel_strike_ix_clusters):
        # Create time axis
        v_acc_time = np.linspace(0.0, len(v_acc)/samp_freq, len(v_acc))
        v_disp_time = np.linspace(0.0, len(v_disp)/samp_freq, len(v_disp))
        # Create axes for plotting
        fig, axs = plt.subplots(5)
        fig.tight_layout()
        # Plot vertical, ML, AP acceleration data
        self.plot_acc_data(v_acc, v_acc_time, 'Vertical Acceleration',
                           heel_strike_ix_clusters, invalid_ix, axs[0])
        self.plot_acc_data(ml_acc, v_acc_time, 'Mediolateral Acceleration',
                           heel_strike_ix_clusters, invalid_ix, axs[1])
        self.plot_acc_data(ap_acc, v_acc_time, 'Anteroposterior Acceleration',
                           heel_strike_ix_clusters, invalid_ix, axs[2])
        # Plot vertical displacement data
        self.plot_acc_data(v_disp, v_disp_time,
                           'Vertical Displacement',
                           heel_strike_ix_clusters, invalid_ix, axs[3])
        # Plot the distribition of the changes in vertical height for COM
        axs[4].title.set_text(
            'Distribution of Vertical Changes of COM Per Step')
        axs[4].set_xlabel('Vertical Changes of COM Per Step (m)')
        axs[4].set_ylabel('Number of Occurrences')
        if pd.DataFrame(com_v_deltas).columns.size > 0:
            y, x, _ = axs[4].hist(com_v_deltas, bins=1000)
            # Add legend w/ descriptive stats of changes in vertical height for COM
            axs[4].text((x.max() - 0.2*x.max()), (y.max() - 0.6*y.max()),
                           pd.DataFrame(com_v_deltas).describe().to_string())
        return fig

    def plot_acc_data(self, acc_data, time, title, heel_ixs, invalid_ixs, ax):
        # Plot vertical acceleration data
        ax.title.set_text(
            f'{title} of COM During Walking Bout')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{title} of COM (m/s^2)')
        # Plot the vertical acceleration signal
        ax.plot(time, acc_data)
        for i in range(len(heel_ixs)):
            v_cluster = heel_ixs[i]
            v_cluster = np.array(v_cluster)
            color = (0, i / 20.0, 0, 1)
            for ix in v_cluster:
                ax.axvline(x=time[ix], color=color, alpha=0.1,
                               ls='--')
        for i in invalid_ixs:
            ax.axvline(x=time[i], color='red', alpha=0.7, ls='--')


# class GaitAnalyzerV2:
#
#     def __init__(self):
#         self.gse_viz = GSEViz()
#
#     def estimate_gait_speed(self, user_data: UserData, hpf, max_com_v_delta, plot_gait_cycles):
#         """
#         Several assumptions are made in this version of the gait speed
#         estimator:
#         1. The orientation of the sensor is such that the vertical axis of the
#             accelerometer is parallel to the vertical axis of the body, i.e. we
#             do not need to transform the accelerometer axes from sensor to
#             global
#         2. The heel strike is defined by the peak anteroposterior acceleration
#             of the stride cycle (Ziljstra 2003)
#         3. The vertical velocity upon each heel strike is approximately zero
#         See Rispens et. al. 2021 for more information on the design of this
#         gait speed estimator.
#         https://www.mdpi.com/1424-8220/21/5/1854/htm#B13-sensors-21-01854
#         :param user_data:
#         :return:
#         """
#         # TODO: refactor this function and the user_data object to have
#         #  segmented walking bouts, this function should assume to have
#         #  walking bouts readily available
#         # Access data required for gait speed estimation from keyword arguments
#         # Get acceleration values, user height, and sampling frequency.
#         lpf_data = user_data.get_imu_data(IMUDataFilterType.LPF)
#         ap_acc_data = lpf_data.get_acc_axis_data('anteroposterior')
#         user_height = user_data.get_clinical_demo_data().get_height()
#         # See Frisancho et al. 2007 for leg length estimation
#         # https://journals.sagepub.com/doi/pdf/10.1177/1545968314532031
#         leg_length = 0.48 * user_height
#         samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
#         # Detect the peaks (heel strikes) in the walking data, defined as peaks in the anteroposterior axis
#         heel_strike_indexes = self._detect_peaks(ap_acc_data)
#         step_start_ixs = heel_strike_indexes[:-1]
#         step_end_ixs = heel_strike_indexes[1:]
#         # Given assumption 1, remove the effects of gravity from the vertical
#         # acc data
#         lpf_v_data = lpf_data.get_acc_axis_data('vertical')
#         v_acc_data = lpf_v_data - np.mean(lpf_v_data)
#         step_lengths, tot_time = self.estimate_step_lengths(
#             v_acc_data, samp_freq, step_start_ixs,
#             step_end_ixs, leg_length, max_com_v_delta, plot_gait_cycles, hpf)
#         total_distance = step_lengths.sum()
#         gait_speed = (total_distance/tot_time)
#         # self.plot_gait_cycles(v_displacement, valid_strike_ixs, invalid_strike_ixs, samp_freq)
#         # self.gse_viz.plot_gse_results(user_data, v_peak_indexes,
#         #                               ap_peak_indexes, v_displacement)
#         return gait_speed
#
#     def _detect_peaks(self, acc_data):
#         height = None
#         threshold = None
#         distance = None
#         prominence = None
#         width = None
#         wlen = None
#         rel_height = 0.5
#         plateau_size = None
#         peaks = PeakDetector().detect_peaks(acc_data, height=height,
#                 threshold=threshold, distance=distance,prominence=prominence,
#                 width=width, wlen=wlen, rel_height=rel_height,
#                                             plateau_size=plateau_size)
#         return peaks
#
#     def estimate_step_lengths(self, v_acc, samp_freq,
#                                step_start_ixs, step_end_ixs, leg_length,
#                                max_com_v_delta, plot_walking_bout, hpf):
#         # Initialize plotting variables
#         valid_strike_ixs = []
#         invalid_strike_ixs = []
#         com_v_deltas = []
#         v_displacement = []
#         # Initialize step lengths in walking bout and total time spent walking
#         tot_time = 0.0
#         step_lengths = []
#         # For every step (interval between ap peak)
#         for start_ix, end_ix in zip(step_start_ixs, step_end_ixs):
#             if (end_ix-start_ix)/samp_freq > 0.43:
#                 # Calculate the vertical displacement of that step
#                 step_v_disp = self.estimate_v_displacement(v_acc, start_ix,
#                                                       end_ix, samp_freq, hpf)
#                 # Add step vertical displacement to the walking bout
#                 # vertical displacment
#                 v_displacement.extend(step_v_disp)
#                 # Compute the difference between the largest and smallest vertical
#                 # displacement of CoM
#                 com_v_delta = max(step_v_disp) - min(step_v_disp)
#                 com_v_deltas.append(com_v_delta)
#                 # Introduce a check to make sure that COM displacement is less than
#                 # an acceptable max value (0.08m = 8cm):
#                 # https://journals.physiology.org/doi/full/10.1152/japplphysiol.00103.2005#:~:text=The%20average%20vertical%20displacement%20of,speeds%20(P%20%3D%200.0001).
#                 # (filters out erroneous COM displacement values)
#                 if com_v_delta < max_com_v_delta:
#                     # Formula for step length derived from inverted pendulum model
#                     step_lengths.append(self._calc_step_length(com_v_delta, leg_length))
#                     # Consider step indices valid
#                     valid_strike_ixs.append(len(v_displacement)-1)
#                     # Increment the total time up
#                     tot_time += ((end_ix - start_ix) / samp_freq)
#                 else:
#                     # Consider the step indices invalid
#                     invalid_strike_ixs.append(len(v_displacement)-1)
#         com_v_deltas = np.array(com_v_deltas)
#         if plot_walking_bout:
#             self.plot_gait_cycles(v_acc, v_displacement, valid_strike_ixs,
#                                   invalid_strike_ixs, samp_freq, com_v_deltas)
#         self._check_step_lengths(step_lengths)
#         return np.array(step_lengths), tot_time
#
#     def plot_gait_cycles(self, v_acc, v_disp, valid_ix, invalid_ix,
#                          samp_freq, com_v_deltas):
#         # Create time axis
#         v_acc_time = np.linspace(0.0, len(v_acc)/samp_freq, len(v_acc))
#         v_disp_time = np.linspace(0.0, len(v_disp)/samp_freq, len(v_disp))
#         # Create axes for plotting
#         fig, axs = plt.subplots(3)
#         fig.tight_layout()
#         axs[0].title.set_text(
#             'Vertical Acceleration of COM During Walking Bout')
#         axs[0].set_xlabel('Time (s)')
#         axs[0].set_ylabel('Vertical Acceleration of COM (m/s^2)')
#         # Plot the vertical acceleration signal
#         axs[0].plot(v_acc_time, v_acc)
#         # Plot the vertical displacement of the COM over time
#         axs[1].plot(v_disp_time, v_disp)
#         # Plot the strike indexes for valid steps
#         axs[1].plot(np.array(v_disp_time)[valid_ix].tolist(),
#                     np.array(v_disp)[valid_ix].tolist(), 'b^')
#         # Plot the strike indexes for invalid steps
#         axs[1].plot(np.array(v_disp_time)[invalid_ix].tolist(),
#                     np.array(v_disp)[invalid_ix].tolist(), 'rv')
#         axs[1].title.set_text('Vertical Displacement of COM During Walking Bout')
#         axs[1].set_xlabel('Time (s)')
#         axs[1].set_ylabel('Vertical Displacement of COM (m)')
#         # Plot the distribition of the changes in vertical height for COM
#         y, x, _ = axs[2].hist(com_v_deltas, bins=1000)
#         # Add legend w/ descriptive stats of changes in vertical height for COM
#         axs[2].text((x.max() - 0.2*x.max()), (y.max() - 0.6*y.max()),
#                        pd.DataFrame(com_v_deltas).describe().to_string())
#         axs[2].title.set_text(
#             'Distribution of Vertical Changes of COM Per Step')
#         axs[2].set_xlabel('Vertical Changes of COM Per Step (m)')
#         axs[2].set_ylabel('Number of Occurrences')
#         plt.show()
#
#     def _calc_step_length(self, v_disp, leg_length):
#         g = ((2 * leg_length - v_disp) * v_disp)
#         step_length = 1.25 * 2 * (g ** 0.5)
#         # step_length = 2 * (g ** 0.5)
#         # Apply correction for mediolateral component of step length
#         # if ((step_length ** 2) > ((0.094 * leg_length) ** 2)):
#         #     step_length = ((step_length ** 2) - ((0.094 * leg_length) ** 2)) ** 0.5
#         return step_length
#
#
#     def _check_step_lengths(self, step_lengths):
#         if np.isnan(step_lengths).any():
#             raise ValueError(f'Computed step lengths contain erroneous value {str(step_lengths)}')
#
#     def estimate_v_displacement(self, v_acc, start_ix,
#                                       end_ix, samp_freq, hpf):
#         period = 1 / samp_freq
#         # The initial position of the CoM at t=0 is arbitrary, set to 0
#         p0 = 0.0
#         # Given assumption 3 of estimate_gait_speed(), we assume initial
#         # vertical velocity at each heel strike to be zero
#         v0 = 0.0
#         acc = v_acc[start_ix:(end_ix - 1)]
#         vel = self._compute_single_integration(acc, period, v0)
#         # TODO: investigate a more appropriate cut-off frequency for the high-pass filter, atm the filter is confounding the results/is not usefult for preventing integration drift
#         if hpf:
#             vel = MotionFilters().apply_lpass_filter(vel, 0.1,
#                                                      samp_freq, 'highpass')
#         pos = self._compute_single_integration(vel[:-1], period, p0)
#         if hpf:
#             pos = MotionFilters().apply_lpass_filter(pos, 0.1,
#                                                      samp_freq, 'highpass')
#         return pos
#
#     def _compute_single_integration(self, data, period, x0):
#         # single integration for time series data is the sum of (the
#         # product of the signal at time t and the sample period) and
#         # (the current integrated value at time t)
#         x = [x0]
#         x_i = x0
#         for i in data:
#             x_t = i * period + x_i
#             x.append(x_t)
#             x_i = x_t
#         return x
#
#     def detect_gait(self, data):
#         """
#         http://www.l3s.de/~anand/tir14/lectures/ws14-tir-foundations-2.pdf
#         :param data:
#         :return:
#         """
#         # Run kamlan filter on the data
#         # Take unbiased vertical acceleration
#         # Perform discrete fourier transform to detect possible periodic data
#         # Run auto-correlation to remove false-positives
#         pass


