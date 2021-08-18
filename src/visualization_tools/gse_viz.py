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

    def plot_gse_results(self, user_data: UserData, gse_idxs: List[int]):
        fig, axes = self.m_viz.plot_motion_data(user_data)
        imu_data = user_data.get_imu_data()[IMUDataFilterType.LPF]
        tri_lin_acc = imu_data.get_triax_acc_data()
        time = imu_data.get_time()
        axes['axis_vertical'].plot(time[gse_idxs], tri_lin_acc['vertical'][gse_idxs],
                                   'rv')
        axes['axis_mediolateral'].plot(time[gse_idxs], tri_lin_acc['mediolateral'][gse_idxs],
                                   'rv')
        axes['axis_anteroposterior'].plot(time[gse_idxs], tri_lin_acc['anteroposterior'][gse_idxs],
                                   'rv')
        self.m_viz.show_plot()
