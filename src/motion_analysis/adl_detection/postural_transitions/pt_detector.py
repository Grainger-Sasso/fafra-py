import time
import math
import copy
import numpy as np
from scipy.optimize import curve_fit
from typing import List
from matplotlib import pyplot as plt

from src.motion_analysis.frequency_analysis.cwt.continuous_wavelet_transform import CWT
from src.motion_analysis.attitude_estimation.attitude_estimator import AttitudeEstimator
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.gait_analysis.gait_analyzer import GaitAnalyzer
from src.dataset_tools.dataset_builders.builder_instances.sisfall_dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class PTDetector:
    def __init__(self, wavelet_name='mexh'):
        """
        Based on CWT technique found in:
        [1] Atrsaei, Arash et al. “Postural transitions detection and
        characterization in healthy and patient populations using a single
        waist sensor.” Journal of neuroengineering and rehabilitation
        vol. 17,1 70. 3 Jun. 2020, doi:10.1186/s12984-020-00692-4
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7271521/
        """
        self.wavelet_name = wavelet_name
        self.cwt = CWT(wavelet_name)

    def detect_pts(self, user_data: UserData, scales,
                   plot_cwt=False, output_dir=None, filename=None):
        """
        See init for reference to methodology used for CWT-based PT detection
        Assumes data has gravity component removed (e.g., naieve mean
        subtraction, attitude estimation)
        :param user_data:
        :return:
        """
        # Initialize output variable
        pt_events = []
        # Get the attitude estimated data vertical acceleration data
        v_acc_data = user_data.get_imu_data(IMUDataFilterType.ATTITUDE_ESTIMATION).get_acc_axis_data('vertical')
        # Get the sampling period from sampling frequency
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        samp_period = 1/samp_freq
        # Apply CWT to data
        cwt_results = self.find_cwt_peaks(v_acc_data, scales, samp_period)
        coeffs, freqs, coeff_sums, cwt_peak_ixs, cwt_peak_values = cwt_results
        time = np.linspace(0.0, len(coeff_sums) * samp_period, len(coeff_sums))
        # Set the duration in samples around the peak to include at PT
        # candidate region, empirically set to 4 seconds [1]
        pt_duration = 4.0 * samp_freq
        # pt_duration = 2.0 * samp_freq
        # Get potential PT candidates as region of data around CWT peaks
        pt_candidates = [[ix-round(pt_duration/2.0), ix+round(pt_duration/2.0)] for ix in cwt_peak_ixs]
        # Check for edge cases on the ends of the acceleration file
        if pt_candidates[0][0] < 0:
            pt_candidates[0][0] = 0
        if pt_candidates[-1][1] > len(v_acc_data)-1:
            pt_candidates[-1][1] = len(v_acc_data)-1
        # Set model fitting threshold, R^2 value, empiracally determined 0.92
        model_fitting_threshold = 0.92
        # Bounds of postural transition elevation change in meters
        min_pt_height_change = 0.20
        max_pt_height_change = 0.60
        # Initialize the vertical displacement, model curve,
        # and fitting coeffiecient arrays
        v_disps = []
        model_curves = []
        fitting_coeffs = []
        # For all PT candidates
        for pt_candidate, cwt_peak_ix in zip(pt_candidates, cwt_peak_ixs):
            # Calculate vertical displacement through double integration of
            # vertical acceleration
            pt_start_ix = pt_candidate[0]
            pt_end_ix = pt_candidate[1]
            # TODO: go back and make sure that the acceleration is in m/s^2 prior to calculating displacement
            v_disp = np.array(GaitAnalyzer().estimate_v_displacement(v_acc_data,
                                                pt_start_ix, pt_end_ix,
                                                samp_freq))
            v_disp_time = np.linspace(0.0, (len(v_disp)-1)*samp_period,
                                      len(v_disp))
            # Fit the vertical displacement data to sigmoid model, returns
            # optimal model params (mp1-mp4) and optimal model param covariance
            mp_opt, mp_cov = curve_fit(
                self._fitting_function, v_disp_time, v_disp)
            # Calculate model fitting coefficient, R^2
            model_curve = self._fitting_function(v_disp_time, mp_opt[0], mp_opt[1],
                                                 mp_opt[2], mp_opt[3])
            model_r_squared = self.calculate_corr_coeff(v_disp, model_curve)
            v_disps.append(v_disp)
            model_curves.append(model_curve)
            fitting_coeffs.append(model_r_squared)

        if plot_cwt:
            fig, axs = self._plot_cwt(user_data, coeffs, freqs,
                                      samp_period, coeff_sums,
                                      cwt_peak_ixs, cwt_peak_values,
                                      output_dir, filename)
            pt_ixs = [item for sublist in pt_candidates
                      for item in sublist]
            axs[1].plot(time[pt_ixs], coeff_sums[pt_ixs], 'yv')
            self.plot_fitted_curves(axs[2], v_disps, model_curves, time, pt_candidates)
            axs[3].plot(time, v_acc_data)
            axs[3].plot(time, np.zeros(len(time)))
            # Plot the vertical acceleration signal
            plt.show()

        # If the fitting coefficient exceeds fitting threshold
        if (model_r_squared > model_fitting_threshold) and (
                min_pt_height_change < abs(mp_opt[1]) < max_pt_height_change):
            # Consider PT candidate to be PT, add to output variable
            pt_index = cwt_peak_ix
            pt_time = mp_opt[2]
            pt_duration = self.calc_transition_duration_a(
                mp_opt[1], mp_opt[3])
            if mp_opt[1] > 0.0:
                pt_type = 'sit-to-stand'
            else:
                pt_type = 'stand-to_sit'
            pt_events.append({'pt_index': pt_index, 'pt_time': pt_time,
                              'pt_duration': pt_duration, 'pt_type': pt_type})
        return pt_events

    def plot_fitted_curves(self, ax, v_disps, model_curves, time, pt_candidates):
        v_disp_curve = np.zeros(len(time))
        whole_model_curve = np.zeros(len(time))
        for pt_candidate, v_disp, model_curve in zip(pt_candidates, v_disps, model_curves):
            pt_start_ix = pt_candidate[0]
            pt_end_ix = pt_candidate[1]
            v_disp_curve[pt_start_ix: pt_end_ix] = v_disp
            whole_model_curve[pt_start_ix: pt_end_ix] = model_curve
        ax.plot(time, v_disp_curve, color='blue')
        ax.plot(time, whole_model_curve, color='red')

    def calc_transition_duration_a(self, mp2, mp4):
        # TODO: better define the tuning factor, article claims it is the
        #  acceleration threshold at start and end of plateau, don't know what
        #  this means
        tuning_factor = 1.0
        beta = ((mp4 ** 2) * tuning_factor)/mp2
        alpha = 2 * np.log((2 * beta) / ((-2 * beta) + 1 - math.sqrt(1 - (4 * beta))))
        transition_duration = alpha * mp4
        return transition_duration

    def calc_transition_duration_omega(self):
        # Calculate the transition duration using angular velocity, omega
        pass

    def calculate_corr_coeff(self, x, y):
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        return r_squared

    def _fitting_function(self, x, mp1, mp2, mp3, mp4):
        """
        Sigmoidal model fitting function
        :param x: Input vector
        :param mp1: Accounts for linear drift
        :param mp2: Determines the amplitude of the elevation change
        :param mp3: Time localization of PT event
        :param mp4: Linearly proportional to the transition duration
        :return:
        """
        return (mp1*x) + (mp2/(1+(np.exp((mp3-x)/mp4))))

    def find_cwt_peaks(self, v_acc_data, scales, samp_period):
        coeffs, freqs = self.cwt.apply_cwt(v_acc_data, scales, samp_period)
        coeff_sums = self.cwt.sum_coeffs(coeffs)
        # Find potential PTs as peaks in the CWT data
        cwt_peaks = self.cwt.detect_cwt_peaks(coeff_sums, samp_period)
        cwt_peak_ixs = cwt_peaks[0]
        cwt_peak_values = cwt_peaks[1]['peak_heights']
        results = [coeffs, freqs, coeff_sums, cwt_peak_ixs, cwt_peak_values]
        return results

    def _plot_cwt(self, user_data, coeffs, freqs, samp_period, coeff_sums,
                  cwt_peak_ixs, cwt_peak_values, output_dir, filename):
        act_code = user_data.get_imu_data(
            IMUDataFilterType.ATTITUDE_ESTIMATION).get_activity_code()
        act_description = user_data.get_imu_data(
            IMUDataFilterType.ATTITUDE_ESTIMATION).get_activity_description()
        fig, axs = self.cwt.plot_cwt_results(coeffs, freqs, samp_period, coeff_sums,
                             cwt_peak_ixs, cwt_peak_values, act_code,
                             act_description, output_dir, filename)
        return fig, axs

def main():
    """
    Function should give me a sense of how the CWT behaves on ADL data
    Want to return a bunch of images of peak detection results
    Look at a couple cases first as a sanity check, then rip the whole batch
    :return:
    """
    t0 = time.time()
    # Set the paths to the sisfall dataset
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\SisFall_csv\SisFall_elderly_csv'
    # Instantiate the dataset builder
    db = DatasetBuilder()
    # Build dataset
    # dataset: Dataset = db.build_dataset(path, '', True, 8.0)
    dataset: Dataset = db.build_dataset(path, '', False, 0.0)
    # Get activity codes of entire dataset
    act_code_data = dataset.get_activity_codes()
    t_dataset = time.time()
    print(str(t_dataset-t0))
    # Get instances of SiSt and StSi
    pt_act_codes = ['D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13']
    adl_dataset: List[UserData] = [data for data in dataset.get_dataset() if data.get_imu_data(IMUDataFilterType.RAW).get_activity_code() in pt_act_codes]
    min_max_scales = [250.0, 25.0]
    num_scales = 100
    scales = np.linspace(min_max_scales[0], min_max_scales[1],
                         num_scales).tolist()
    output_dir = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\cwt\plots'
    # Iterate through the ADL dataset and run CWT + peak detection on the batch
    pt_detector = PTDetector()
    adl_dataset_pt_events = []
    for user_data in adl_dataset:
        filename = user_data.get_imu_data_file_name()
        # print(data.get_imu_data().get_activity_code())
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        preprocess_data(user_data, samp_freq, 'mean_subtraction')
        pt_events = pt_detector.detect_pts(user_data, scales,
                               plot_cwt=False, output_dir=None, filename=None)
        adl_dataset_pt_events.append(pt_events)

def preprocess_data(user_data: UserData, samp_freq, type: str):
    # Apply LPF
    lpf_data(user_data, samp_freq)
    if type == 'mean_subtraction':
        v_acc_data = user_data.get_imu_data(IMUDataFilterType.LPF).get_acc_axis_data('vertical')
        v_acc_data = v_acc_data - np.mean(v_acc_data)
    elif type == 'remove_gravity':
        # Run the attitude estimation and remove the gravity componenet from
        v_acc_data = remove_gravity(user_data)
    else:
        raise ValueError(f'Invalid preprocessing type: {type}')
    normalized_imu_data = copy.deepcopy(user_data.get_imu_data(IMUDataFilterType.LPF))
    normalized_imu_data.set_acc_axis_data('vertical', v_acc_data)
    user_data.add_filtered_data(normalized_imu_data,
                                IMUDataFilterType.ATTITUDE_ESTIMATION)

def lpf_data(user_data: UserData, samp_freq):
    mf = MotionFilters()
    imu_data: IMUData = user_data.get_imu_data(IMUDataFilterType.RAW)
    act_code = imu_data.get_activity_code()
    act_description = imu_data.get_activity_description()
    all_raw_data = imu_data.get_all_data()
    lpf_data_all_axis = []
    for data in all_raw_data:
        lpf_data_all_axis.append(
            mf.apply_lpass_filter(data, 2, samp_freq))
    lpf_imu_data = generate_imu_data_instance(
        lpf_data_all_axis, samp_freq, act_code, act_description)
    user_data.add_filtered_data(lpf_imu_data, IMUDataFilterType.LPF)

def generate_imu_data_instance(data, samp_freq, act_code, act_description):
    v_acc_data = np.array(data[0])
    ml_acc_data = np.array(data[1])
    ap_acc_data = np.array(data[2])
    yaw_gyr_data = np.array(data[3])
    pitch_gyr_data = np.array(data[4])
    roll_gyr_data = np.array(data[5])
    time = np.linspace(0, len(v_acc_data) / int(samp_freq),
                       len(v_acc_data))
    return IMUData(act_code, act_description, v_acc_data, ml_acc_data,
                ap_acc_data, yaw_gyr_data, pitch_gyr_data, roll_gyr_data, time)

def remove_gravity(user_data):
    att_est = AttitudeEstimator()
    # Estimate the angle between the z-axis and the x-y plane
    theta = att_est.estimate_attitude(user_data, False)
    # Use the angle estimation to remove the effects of gravity from the
    # vertical acceleration
    v_acc_data = user_data.get_imu_data()[
        IMUDataFilterType.LPF].get_acc_axis_data('vertical')
    # As theta approaches 90°, amount of gravity component removed increases
    # As theta approaches 0°, amount of gravity component removed decreases
    # TODO: Implement the cos of the thesta
    rm_g_v_acc_data = [v_acc - (9.8*theta) for v_acc, theta in zip(v_acc_data, theta)]
    return rm_g_v_acc_data




if __name__ == "__main__":
    main()