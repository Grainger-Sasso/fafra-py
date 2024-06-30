import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.visualization_tools.motion_visualizer import MotionVisualizer


class GSEViz:
    def __init__(self):
        self.m_viz = MotionVisualizer()

    def plot_gse_results(self, user_data: UserData,
                         v_peak_indexes: List[int],
                         ap_peak_indexes: List[int],
                         displacement):
        acc_fig, acc_axes = self.m_viz.plot_acceleration_data(user_data)
        gyr_fig, gyr_axes = self.m_viz.plot_gyroscope_data(user_data)
        imu_data = user_data.get_imu_data(IMUDataFilterType.LPF)
        tri_lin_acc = imu_data.get_triax_acc_data()
        time = imu_data.get_time()
        acc_axes['axis_vertical'].plot(time[v_peak_indexes],
                                   tri_lin_acc['vertical'][v_peak_indexes],
                                   'rv')
        acc_axes['axis_anteroposterior'].plot(time[v_peak_indexes],
                                          tri_lin_acc['anteroposterior'][
                                              v_peak_indexes],
                                          'rv')
        acc_axes['axis_anteroposterior'].plot(time[ap_peak_indexes],
                                          tri_lin_acc['anteroposterior'][
                                              ap_peak_indexes],
                                          'bo')
        self.plot_displacement(displacement, time, ap_peak_indexes)
        self.m_viz.show_plot()

    def plot_displacement(self, displacement, time, ap_peak_ixs):
        fig, ax = plt.subplots(1, sharex=True)
        ax.plot(time[ap_peak_ixs[0]:ap_peak_ixs[-1]], displacement, color='red')
        for ix in ap_peak_ixs:
            ax.axvline(x=time[ix])

