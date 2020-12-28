import numpy as np
import scipy
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.dataset_tools.motion_data.acceleration.acceleration import Acceleration
from src.motion_analysis.filters.motion_filters import MotionFilters


class FallDetector:

    def __init__(self):
        self.detection_methods = ['sucerquia']
        self.motion_filters = MotionFilters()

    def detect_falls(self, motion_dataset: MotionDataset, method: str):
        if method == 'sucerquia' or 'Sucerquia':
            self.__detect_falls_sucerquia(motion_dataset)
        else:
            raise ValueError(f'Fall detection method provided, {method}, not available')

    def __detect_falls_sucerquia(self, motion_dataset: MotionDataset):
        sampling_rate = motion_dataset.get_sampling_rate()
        one_s_window_size = int(sampling_rate * 1)
        number_activities = len(motion_dataset.get_motion_data())
        # Array of boolean values indicating a fall occurrance for every motion data activity in the motion dataset
        fall_occurred = np.zeros(number_activities, dtype=bool)
        # Array of floating point recording fall times (if they occurred) of motion data activities, otherwise nan
        fall_times = np.empty(number_activities)
        fall_times[:] = np.nan
        # Apply low pass filter
        motion_dataset.apply_lp_filter()
        # Apply derivative, feeds into metric J1
        motion_dataset.calculate_first_derivative_data()
        # Apply Kalman filter, feeds into metric J2
        motion_dataset.apply_kalman_filter()
        # Initialize the results modifier index
        fall_detection_index = 0
        # Perform fall detection on every motion data activity
        for motion_data in motion_dataset.get_motion_data():
            fall_detected = False
            fall_time = np.nan
            for tri_lin_acc in motion_data.get_tri_lin_accs():
                # Get all axes in triaxial acceleration
                x_ax = tri_lin_acc.get_x_axis()
                y_ax = tri_lin_acc.get_y_axis()
                z_ax = tri_lin_acc.get_z_axis()
                # Calculate metric J1: J1[k] = RMS(d(a[k])/dt)
                metric_j1 = self.__calculate_metric_j1(x_ax, y_ax, z_ax)
                # Calculate metric J2: J2[k] = std(x[k]) for a 1s sliding window for each time step k of the first 3 values of Kalman filtered data
                metric_j2 = self.__calculate_metric_j2(x_ax, y_ax, z_ax, one_s_window_size)
                metric_j3 = self.__calculate_metric_j3(metric_j1, metric_j2, one_s_window_size)
                print('yippee')

        # Calculate metric J3: J1 * (J2)^2
        # For all times, t0-tf
        # If score > threshold
        # If data is periodic for 3 second window after potential fall candidate (from unbiased, Kalman filtered vertical axis data)
        # A fall has occurred, label the time
        return fall_occurred, fall_times


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
        # Get window size equal to 1 second of recording (sampling rate * 1s)

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

    def __pad_j2_std_data(self):
        pass

    # TODO: Make a generic FallDetector class, make this class a child
    def __perform_fall_detection_on_window(self):
        pass

