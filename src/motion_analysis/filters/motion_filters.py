import numpy as np
import itertools
from scipy.ndimage import generic_filter
from scipy import signal
from filterpy.kalman import KalmanFilter


class MotionFilters:

    def __init__(self):
        self.kalman_filter: KalmanFilter = KalmanFilter(dim_x=4, dim_z=4)

    def apply_moving_average(self, data: np.array, n=3):
        return np.convolve(data, np.ones(n), 'valid') / n

    def apply_lpass_filter(self, data: np.array, cutoff_freq,
                           samp_freq, high_low_pass='lowpass'):
        # Parameters for Butterworth filter found (3.1. Pre-Processing Stage):
        # https://www.mdpi.com/1424-8220/18/4/1101
        # Create a 4th order lowpass butterworth filter
        cutoff_freq = (cutoff_freq/(0.5*samp_freq))
        b, a = signal.butter(4, cutoff_freq, high_low_pass, analog=False)
        # Apply filter to input data
        return np.array(signal.filtfilt(b, a, data), dtype=np.float16)
        #
        # return np.array(signal.filtfilt(b, a, data))

    def downsample_data(self, current_data, current_sampling_rate, new_sampling_rate):
        current_num_samples = len(current_data)
        duration = current_num_samples/current_sampling_rate
        new_num_samples = int(duration * new_sampling_rate)
        sampling_indices = np.linspace(0, current_num_samples-1, new_num_samples, dtype=int)
        return current_data[sampling_indices]

    def apply_kalman_filter(self, x_ml_ax, y_v_ax, z_ap_ax, sampling_rate):
        # TODO: evaluate run-time and run-time optimization strategies
        # Filter design based off of model reported here:
        # https://www.mdpi.com/1424-8220/18/4/1101
        # Initial conditions (output vextor is [ax, ay, az, ay-bay] where bay is current sensor bias)
        bay_window = np.array([-1.0])
        all_bays = []
        # Max number of elements in sliding window for vertical bias (1s worth of readings)
        max_bay_window_size = sampling_rate * 1.0
        x0 = np.array([[0.0], [-1.0], [0.0], [0.0]])
        self.__initialize_kalman_filter(x0)
        kf_filtered_data = []
        for x_ax_lp_val, y_ax_lp_val, z_ax_lp_val in zip(x_ml_ax, y_v_ax, z_ap_ax):
            bay = np.mean(bay_window)
            all_bays.append(bay)
            measurement = np.array([x_ax_lp_val, y_ax_lp_val, z_ax_lp_val, y_ax_lp_val-bay])
            self.kalman_filter.predict()
            self.kalman_filter.update(measurement)
            kf_filtered_data.append(self.kalman_filter.x)
            # bay_window = self.__update_vertical_bias_window(bay_window, self.kalman_filter.x[0:3], max_bay_window_size)
            bay_window = self.__update_vertical_bias_window(bay_window, self.kalman_filter.x[1], max_bay_window_size)
        len_kf_data = len(kf_filtered_data)
        x_kf_data = np.zeros(len_kf_data)
        y_kf_data = np.zeros(len_kf_data)
        z_kf_data = np.zeros(len_kf_data)
        unbiased_y_kf_data = np.zeros(len_kf_data)
        for ix, kf_data in enumerate(kf_filtered_data):
            x_kf_data[ix] = kf_data[0][0]
            y_kf_data[ix] = kf_data[1][0]
            z_kf_data[ix] = kf_data[2][0]
            unbiased_y_kf_data[ix] = kf_data[3][0]
        return x_kf_data, y_kf_data, z_kf_data, unbiased_y_kf_data

    def __update_vertical_bias_window(self, bay_win, kf_measurement, max_bay_window_size):
        window_size = len(bay_win)
        avg_measurement = np.average(kf_measurement)
        if window_size == max_bay_window_size:
            # Append measurement to bay window to front of window
            np.put(bay_win, 0, avg_measurement)
            # Move first element to end of list, shift all other values left one position
            new_bay = np.roll(bay_win, -1)
        else:
            # Append measurement to bay window
            new_bay = np.append(bay_win, avg_measurement)
        return new_bay

    def __initialize_kalman_filter(self, x0):
        # Filter params, variable names follow Kalman filter param conventions
        # State dynamics matrix, set to Identity matrix
        A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=float)
        # Sensor matrix, set to Identity matrix
        C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=float)
        # Process noise covariance matrix
        Q = np.array([[1e-6, 0.0, 0.0, 0.0], [0.0, 1e-6, 0.0, 0.0], [0.0, 0.0, 1e-6, 0.0], [0.0, 0.0, 0.0, 1e-6]],
                     dtype=float)
        # Sensor noise covariance matrix
        R = np.array(
            [[2.5e-3, 0.0, 0.0, 0.0], [0.0, 2.5e-3, 0.0, 0.0], [0.0, 0.0, 2.5e-3, 0.0], [0.0, 0.0, 0.0, 1e-4]],
            dtype=float)
        # Initialize value of the state vector
        self.kalman_filter.x = x0
        # Initialize covariance matrix
        self.kalman_filter.P = Q
        # Set state dynamics matrix
        self.kalman_filter.F = A
        # Set sensor matrix
        self.kalman_filter.H = C
        # Set the sensor noise covariance matrix
        self.kalman_filter.R = R
        # Set the process noise covariance matrix
        self.kalman_filter.Q = Q

    def calculate_first_derivative(self, x, y):
        """
        Calculates d(x)/dy
        :param x: one dimensional data
        :param y: one dimensional data
        :return: d(x)/dy
        """
        # Divide dx by dy
        return np.array([dx/dy for dx, dy in zip(np.diff(x), np.diff(y))])

    def calculate_triaxial_rms(self, x: np.array, y: np.array, z: np.array):
        # rms = sqrt((1/n)(a^2+b^2+c^2)) where n=3
        rms_matrix = np.array((x, y, z))
        return np.sqrt(np.mean(np.power(rms_matrix, 2), axis=0))

    def calculate_resultant_vector(self, x: np.array, y: np.array, z: np.array):
        rms_matrix = np.array((x, y, z))
        return np.sqrt(np.sum(np.power(rms_matrix, 2), axis=0))

    def calculate_rms(self, x):
        return np.sqrt(np.mean(np.power(x, 2), axis=0))

    def generic_filter_sliding_std_dev(self, data, window_size):
        return generic_filter(data, np.std, size=window_size)

    def generic_filter_max(self, data, window_size):
        return generic_filter(data, np.max, size=window_size)

    def windowed_sum(self, a, win):
        table = np.cumsum(np.cumsum(a, axis=0), axis=1)
        win_sum = np.empty(tuple(np.subtract(a.shape, win - 1)))
        win_sum[0, 0] = table[win - 1, win - 1]
        win_sum[0, 1:] = table[win - 1, win:] - table[win - 1, :-win]
        win_sum[1:, 0] = table[win:, win - 1] - table[:-win, win - 1]
        win_sum[1:, 1:] = (table[win:, win:] + table[:-win, :-win] -
                           table[win:, :-win] - table[:-win, win:])
        return win_sum

    def strided_sliding_std_dev(self, data, radius):
        windowed = self.__rolling_window(data, radius)
        shape = windowed.shape
        windowed = windowed.reshape(shape[0], shape[1], -1)
        return windowed.std(axis=-1)

    def __rolling_window(self, a, window):
        """Takes a numpy array *a* and a sequence of (or single) *window* lengths
        and returns a view of *a* that represents a moving window."""
        if not hasattr(window, '__iter__'):
            return self.__rolling_window_lastaxis(a, window)
        for i, win in enumerate(window):
            if win > 1:
                a = a.swapaxes(i, -1)
                a = self.__rolling_window_lastaxis(a, win)
                a = a.swapaxes(-2, i)
        return a

    def __rolling_window_lastaxis(self, a, window):
        """Directly taken from Erik Rigtorp's post to numpy-discussion.
        <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
        if window < 1:
            raise ValueError
        if window > a.shape[-1]:
            raise ValueError
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def unit_vector_norm(self, x: np.array):
        return x / np.linalg.norm(x)

    def __pairwise(self, iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    # def kalman_filter(self, motion_data: MotionData):
    #     # Filter design based off of model reported here:
    #     # https://www.mdpi.com/1424-8220/18/4/1101
    #     for tri_ax_acc in motion_data.get_tri_lin_accs():
    #         x_ax = tri_ax_acc.get_x_axis()
    #         y_ax = tri_ax_acc.get_y_axis()
    #         z_ax = tri_ax_acc.get_z_axis()
    #         # Filter params, variable names follow Kalman filter param conventions
    #         # State dynamics matrix, set to Identity matrix
    #         A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    #         # Sensor matrix, set to Identity matrix
    #         C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    #         # Process noise covariance matrix
    #         Q = np.array([[1e-6, 0.0, 0.0, 0.0], [0.0, 1e-6, 0.0, 0.0], [0.0, 0.0, 1e-6, 0.0], [0.0, 0.0, 0.0, 1e-6]], dtype=float)
    #         # Sensor noise covariance matrix
    #         R = np.array([[2.5e-3, 0.0, 0.0, 0.0], [0.0, 2.5e-3, 0.0, 0.0], [0.0, 0.0, 2.5e-3, 0.0], [0.0, 0.0, 0.0, 1e-4]], dtype=float)
    #         # Initial conditions (output vextor is [ax, ay, az, ay-bay] where bay is current sensor bias)
    #         bay = -1.0
    #         x0 = np.array([[0.0], [-1.0], [0.0], [0.0]])
    #         kf = self.__initialize_kalman_filter(x0, Q, A, C, R)
    #         kf_filtered_data = []
    #         for ix in range(len(x_ax.get_acceleration_data())-1):
    #             measurement = np.array([x_ax.get_lp_filtered_data()[ix], y_ax.get_lp_filtered_data()[ix],
    #                                     z_ax.get_lp_filtered_data()[ix], y_ax.get_lp_filtered_data()[ix]-bay])
    #             kf.predict()
    #             kf.update(measurement)
    #             kf_filtered_data.append(kf.x)
    #             # TODO: Replace this bias definition with a 1s sliding window kf y-axis data
    #             bay = np.average(kf.x)
    #         x_kf_data = []
    #         y_kf_data = []
    #         z_kf_data = []
    #         x_kf_data.append(x0[0][0])
    #         y_kf_data.append(x0[1][0])
    #         z_kf_data.append(x0[2][0])
    #         for kf_data in kf_filtered_data:
    #             x_kf_data.append(kf_data[0][0])
    #             y_kf_data.append(kf_data[1][0])
    #             z_kf_data.append(kf_data[2][0])
    #         x_ax.set_kf_filtered_data(np.array(x_kf_data))
    #         y_ax.set_kf_filtered_data(np.array(y_kf_data))
    #         z_ax.set_kf_filtered_data(np.array(z_kf_data))


