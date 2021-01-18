import numpy as np
import pandas as pd
import os
import random
import string
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.dataset_tools.params.motion_data import MotionData
from src.dataset_tools.motion_data.acceleration.triaxial_acceleration import TriaxialAcceleration
from src.dataset_tools.motion_data.acceleration.acceleration import Acceleration
from src.motion_analysis.filters.motion_filters import MotionFilters


class SucerquiaFallDetector:

    def __init__(self):
        self.motion_filters = MotionFilters()
        self.sucerquia_metric_threshold: float = 4.0
        self.sucerquia_periodicity_threshold: float = 1.5

    def detect_falls_in_motion_dataset(self, motion_dataset: MotionDataset, write_results_to_csv=False, output_path=''):
        sampling_rate = motion_dataset.get_sampling_rate()
        number_activities = len(motion_dataset.get_motion_data())
        # Array of boolean values indicating a fall occurrance for every motion data activity in the motion dataset
        ds_fall_measurements = np.zeros(number_activities, dtype=bool)
        ds_fall_predictions = np.zeros(number_activities, dtype=bool)
        # Measurement-prediction comparison
        ds_mp_comparison = np.zeros(number_activities, dtype=bool)
        # Array of floating point recording fall times (if they occurred) of motion data activities, otherwise nan
        ds_fall_indices = [np.nan] * number_activities
        # Apply low pass filter
        motion_dataset.apply_lp_filter()
        # Apply derivative, feeds into metric J1
        motion_dataset.calculate_first_derivative_data()
        # Apply Kalman filter, feeds into metric J2
        motion_dataset.apply_kalman_filter()
        # Perform fall detection on every motion data activity
        for dataset_index, motion_data in enumerate(motion_dataset.get_motion_data()):
            md_fall_measurement, md_fall_predictions, md_mp_comparison, md_fall_index = self.detect_falls_in_motion_data(motion_data, sampling_rate, True)
            ds_fall_measurements[dataset_index] = md_fall_measurement
            ds_fall_predictions[dataset_index] = md_fall_predictions
            ds_mp_comparison[dataset_index] = md_mp_comparison
            ds_fall_indices[dataset_index] = md_fall_index
        results_df = pd.DataFrame({"measurements": ds_fall_measurements, "predictions": ds_fall_predictions,
                                   "comparison": ds_mp_comparison, "indices": np.array(ds_fall_indices)})
        results_df = self.__add_confustion_metrics_to_ds_results(results_df, number_activities)
        if write_results_to_csv:
            # Write results to csv
            self.__write_dataset_results_to_csv(results_df, output_path, motion_dataset.get_name())
        return results_df

    def detect_falls_in_motion_data(self, motion_data: MotionData, sampling_rate: float, filter_flag: bool):
        if not filter_flag:
            motion_data.apply_lp_filter(sampling_rate)
            motion_data.calculate_first_derivative_data()
            motion_data.apply_kalman_filter(sampling_rate)
        # Initialize the fall detection output to indicate fall was not detected
        md_fall_measurement = motion_data.get_activity().get_fall()
        md_fall_prediction = False
        md_fall_index = np.nan
        for tri_lin_acc in motion_data.get_tri_lin_accs():
            triaxial_fall_detected, triaxial_fall_index = self.__detect_falls_in_triaxial_acc(tri_lin_acc,
                                                                                              sampling_rate)
            if triaxial_fall_detected:
                md_fall_prediction = triaxial_fall_detected
                md_fall_index = triaxial_fall_index
                break
        md_mp_comparison = md_fall_measurement == md_fall_prediction
        return md_fall_measurement, md_fall_prediction, md_mp_comparison, md_fall_index

    def __add_confustion_metrics_to_ds_results(self, results_df, number_activities):
        num_correct_arr = [np.nan] * number_activities
        num_correct = np.count_nonzero(results_df['comparison'])
        num_correct_arr[0] = num_correct
        num_incorrect_arr = [np.nan] * number_activities
        num_incorrect = len(results_df['comparison']) - num_correct
        num_incorrect_arr[0] = num_incorrect
        error_rate_arr = [np.nan] * number_activities
        error_rate = num_incorrect / len(results_df['comparison'])
        error_rate_arr[0] = error_rate
        results_df['num_correct'] = num_correct_arr
        results_df['num_incorrect'] = num_incorrect_arr
        results_df['error_rate'] = error_rate_arr
        return results_df
    def __write_dataset_results_to_csv(self, results_df: pd.DataFrame, output_directory: str, dataset_name: str):
        random_alphanumeric = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        output_file_name = 'results_' + dataset_name + '_' + random_alphanumeric + '.csv'
        full_output_path = os.path.join(output_directory, output_file_name)
        results_df.to_csv(full_output_path)

    def __detect_falls_in_triaxial_acc(self, triaxial_acc: TriaxialAcceleration, sampling_rate: float):
        # Initialize output of fall detection to indicate not fall was detected
        triaxial_fall_detected = False
        triaxial_fall_indices = []
        triaxial_fall_index = np.nan
        metric_j1, metric_j2, metric_j3 = self.__calculate_triaxial_metrics(triaxial_acc, sampling_rate)
        # Check if values in array exceed dummy threshold value of 0.14
        # metric_j3_local_max_ind = np.array(argrelextrema(metric_j3, np.greater)[0])
        # local_max_above_thold_mask = np.array(
        #     [val > self.sucerquia_metric_threshold for val in metric_j3[metric_j3_local_max_ind]])
        # above_thold_ind = metric_j3_local_max_ind[local_max_above_thold_mask]
        above_thold_ind = np.array(find_peaks(metric_j3, height=self.sucerquia_metric_threshold)[0])
        if above_thold_ind.size > 0:
            y_ax_unbiased_kf_data = triaxial_acc.get_y_axis().get_unbiased_kf_filtered_data()
            # for every value in above_thold_ind
            for index in above_thold_ind:
                # Get 3s window of data from unbiased y_ax kf data
                window_ix = int((sampling_rate * 3.0) + index)
                if window_ix > len(y_ax_unbiased_kf_data) - 1:
                    window_ix = int(len(y_ax_unbiased_kf_data) - 1)
                periodic_data = y_ax_unbiased_kf_data[index:window_ix]
                # Check the periodicity of the 3s window of data
                periodicity = self.__detect_periodicity(periodic_data, sampling_rate)
                # If the data is not periodic (any period between zero-crossings is too large to be periodic
                # or no zero-crossing occurs), then add this index to the list of fall indices
                if (periodicity > self.sucerquia_periodicity_threshold).any() or len(periodicity) == 0:
                    # This defines a fall (data above metric threshold and data after is not periodic), add the
                    # data to the fall index list
                    triaxial_fall_detected = True
                    triaxial_fall_indices.append(int(index))
        # If there are fall indices detected for this activity
        if triaxial_fall_indices:
            # Set the fall index to the largest fall index detected (value of metric j3)
            triaxial_fall_index = sorted(triaxial_fall_indices, key=lambda ix: metric_j3[ix])[-1]
        return triaxial_fall_detected, triaxial_fall_index

    def __calculate_triaxial_metrics(self, triaxial_acc: TriaxialAcceleration, sampling_rate: float):
        one_s_window_size = int(sampling_rate * 1)
        # Get all axes in triaxial acceleration
        x_ax = triaxial_acc.get_x_axis()
        y_ax = triaxial_acc.get_y_axis()
        z_ax = triaxial_acc.get_z_axis()
        # Calculate metric J1: J1[k] = RMS(d(a[k])/dt)
        metric_j1 = self.__calculate_metric_j1(x_ax, y_ax, z_ax)
        # Calculate metric J2: J2[k] = std(x[k]) for a 1s sliding window for each time step k of the first 3
        # values of Kalman filtered data
        metric_j2 = self.__calculate_metric_j2(x_ax, y_ax, z_ax, one_s_window_size)
        metric_j3 = self.__calculate_metric_j3(metric_j1, metric_j2, one_s_window_size)
        return metric_j1, metric_j2, metric_j3

    def __detect_periodicity(self, data, sampling_rate):
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        periodicity = np.array([(j - i) * 2 / sampling_rate for i, j in zip(zero_crossings, zero_crossings[1:])])
        return periodicity

    def __calculate_metric_j3(self, metric_j1, metric_j2, one_s_window_size):
        j1_sliding_max = np.array(self.motion_filters.generic_filter_max(metric_j1, one_s_window_size))
        j2_sliding_max = np.array(self.motion_filters.generic_filter_max(metric_j2, one_s_window_size))
        j2_sliding_max = np.power(j2_sliding_max, 2)
        metric_j3 = j1_sliding_max * j2_sliding_max
        return metric_j3

    def __calculate_metric_j2(self, x_ax: Acceleration, y_ax: Acceleration, z_ax: Acceleration, one_s_window_size):
        x_kf_data = x_ax.get_kf_filtered_data()
        y_kf_data = y_ax.get_kf_filtered_data()
        z_kf_data = z_ax.get_kf_filtered_data()
        xyz_matrix = np.array((x_kf_data, y_kf_data, z_kf_data))
        x_kf_std = self.motion_filters.generic_filter_sliding_std_dev(x_kf_data, one_s_window_size)
        y_kf_std = self.motion_filters.generic_filter_sliding_std_dev(y_kf_data, one_s_window_size)
        z_kf_std = self.motion_filters.generic_filter_sliding_std_dev(z_kf_data, one_s_window_size)
        metric_j2 = self.motion_filters.calculate_triaxial_rms(x_kf_std, y_kf_std, z_kf_std)
        return metric_j2[:-1]

    def __calculate_metric_j1(self, x_ax: Acceleration, y_ax: Acceleration, z_ax: Acceleration):
        x_der_ax = x_ax.get_first_derivative_data()
        y_der_ax = y_ax.get_first_derivative_data()
        z_der_ax = z_ax.get_first_derivative_data()
        metric_j1 = self.motion_filters.calculate_triaxial_rms(x_der_ax, y_der_ax, z_der_ax)
        return metric_j1
