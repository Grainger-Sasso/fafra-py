import pandas as pd
import numpy as np
from scipy import signal
from typing import List
from src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
from src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration
from src.dataset_tools.params.subject import Subject
from src.dataset_tools.params.activity import Activity
# from src.motion_analysis.filters.motion_filters import MotionFilters


class MotionData:

    def __init__(self, subject: Subject, activity: Activity, trial: str,
                 motion_df: pd.DataFrame, tri_lin_accs: List[TriaxialLinearAcceleration],
                 tri_ang_accs: List[TriaxialAngularAcceleration]):
        self.subject: Subject = subject
        self.activity: Activity = activity
        self.trial: str = trial
        self.motion_df: np.ndarray = motion_df
        self.tri_lin_accs: List[TriaxialLinearAcceleration] = tri_lin_accs
        self.tri_ang_accs: List[TriaxialAngularAcceleration] = tri_ang_accs

    def get_subject(self):
        return self.subject

    def get_activity(self):
        return self.activity

    def get_trial(self):
        return self.trial

    def get_motion_df(self):
        return self.motion_df

    def get_tri_lin_accs(self) -> List[TriaxialLinearAcceleration]:
        return self.tri_lin_accs

    def get_tri_ang_accs(self) -> List[TriaxialAngularAcceleration]:
        return self.tri_ang_accs

    def get_triaxial_accs(self):
        # Flattens list of lists
        accs = [self.tri_lin_accs, self.tri_ang_accs]
        return [acc for sublist in accs for acc in sublist]

    # def calculate_first_derivative_data(self):
    #     filters = MotionFilters()
    #     for tri_acc in self.get_triaxial_accs():
    #         for acc in tri_acc.get_all_axes():
    #             acc.set_first_derivative_data(filters.calculate_first_derivative(acc.acceleration_data, acc.time))
    #
    # def apply_lp_filter(self, sampling_rate):
    #     filters = MotionFilters()
    #     for tri_acc in self.get_triaxial_accs():
    #         for acc in tri_acc.get_all_axes():
    #             acc.set_lp_filtered_data(filters.apply_lpass_filter(acc.acceleration_data, sampling_rate))
    #
    # def apply_kalman_filter(self, sampling_rate):
    #     filters = MotionFilters()
    #     for tri_acc in self.get_triaxial_accs():
    #         x_ax = tri_acc.get_x_axis()
    #         y_ax = tri_acc.get_y_axis()
    #         z_ax = tri_acc.get_z_axis()
    #         x_kf_filtered_data, y_kf_filtered_data, z_kf_filtered_data, unbiased_y_ax_kf_data = filters.apply_kalman_filter(x_ax, y_ax, z_ax, sampling_rate)
    #         x_ax.set_kf_filtered_data(x_kf_filtered_data)
    #         y_ax.set_kf_filtered_data(y_kf_filtered_data)
    #         z_ax.set_kf_filtered_data(z_kf_filtered_data)
    #         y_ax.set_unbiased_kf_filtered_data(unbiased_y_ax_kf_data)
    #
    # def downsample(self, old_sampling_rate, new_sampling_rate):
    #     if old_sampling_rate <= new_sampling_rate:
    #         raise ValueError(f'Sampling rate of {old_sampling_rate} cannot be downsampled to {new_sampling_rate}')
    #
    #     #TODO: add in a revision to the motion_df to account for the downsampling
    #     for tri_acc in self.get_triaxial_accs():
    #         x_ax = tri_acc.get_x_axis()
    #         y_ax = tri_acc.get_y_axis()
    #         z_ax = tri_acc.get_z_axis()
    #         x_ax.set_acceleration_data(self.__downsample_axis(x_ax, old_sampling_rate, new_sampling_rate))
    #         y_ax.set_acceleration_data(self.__downsample_axis(y_ax, old_sampling_rate, new_sampling_rate))
    #         z_ax.set_acceleration_data(self.__downsample_axis(z_ax, old_sampling_rate, new_sampling_rate))
    #
    # def __downsample_axis(self, ax, old_sampling_rate, new_sampling_rate):
    #     old_num_samples = len(ax.get_acceleration_data())
    #     new_num_samples = int((old_num_samples / old_sampling_rate) * new_sampling_rate)
    #     return np.array(signal.resample(ax.get_acceleration_data(), new_num_samples))





