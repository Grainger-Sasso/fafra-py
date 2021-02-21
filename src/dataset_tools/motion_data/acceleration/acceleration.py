import numpy as np
from src.dataset_tools.params.sensor import Sensor



class Acceleration:

    def __init__(self, axis: str, anatomical_axis: str, acc_data: np.array, time, sensor: Sensor):
        self.axis = axis
        self.anatomical_axis = anatomical_axis
        self.acceleration_data = acc_data
        self.lp_filtered_data: np.array = np.zeros(len(acc_data))
        self.kf_filtered_data: np.array = np.zeros(len(acc_data))
        self.unbiased_kf_filtered_data: np.array = np.zeros(len(acc_data))
        self.first_derivative_data: np.array = np.zeros(len(acc_data))
        self.time = time
        self.sensor: Sensor = sensor


    def get_axis(self):
        return self.axis

    def get_anatomical_axis(self):
        return self.anatomical_axis

    def get_acceleration_data(self):
        return self.acceleration_data

    def get_lp_filtered_data(self):
        return self.lp_filtered_data

    def get_kf_filtered_data(self):
        return self.kf_filtered_data

    def get_unbiased_kf_filtered_data(self):
        return self.unbiased_kf_filtered_data

    def get_first_derivative_data(self):
        return self.first_derivative_data

    def get_time(self):
        return self.time

    def get_sensor(self):
        return self.sensor

    def set_acceleration_data(self, data: np.array):
        self.acceleration_data = data

    def set_lp_filtered_data(self, data: np.array):
        self.lp_filtered_data = data

    def set_kf_filtered_data(self, data: np.array):
        self.kf_filtered_data = data

    def set_unbiased_kf_filtered_data(self, data: np.array):
        self.unbiased_kf_filtered_data = data

    def set_first_derivative_data(self, data: np.array):
        self.first_derivative_data = data

    def apply_conversion(self, conversion_method: classmethod):
        self.acceleration_data = conversion_method(self.acceleration_data)

