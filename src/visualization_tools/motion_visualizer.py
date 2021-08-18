import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class MotionVisualizer:
    def plot_motion_data(self, user_data: UserData):
        colors = {IMUDataFilterType.RAW: 'cornflowerblue',
                  IMUDataFilterType.LPF: 'red',
                  IMUDataFilterType.KF: 'yellow'}
        fig, (ax_v, ax_ml, ax_ap) = self._generate_fig_axes()
        axes = {'axis_vertical': ax_v,
                'axis_mediolateral': ax_ml,
                'axis_anteroposterior': ax_ap}
        for filt_type, imu_data in user_data.get_imu_data().items():
            tri_lin_acc = imu_data.get_triax_acc_data()
            time = imu_data.get_time()
            self._plot_triaxial_acc(tri_lin_acc, time, filt_type, axes, colors)
        return fig, axes

    def _generate_fig_axes(self):
        return plt.subplots(3, sharex=True)

    def _plot_triaxial_acc(self, tri_acc, time, filt_type, axes, colors):
        axes['axis_vertical'].grid(True, 'both')
        axes['axis_mediolateral'].grid(True, 'both')
        axes['axis_anteroposterior'].grid(True, 'both')
        self._plot_single_axis(axes['axis_vertical'], tri_acc['vertical'],
                               time, 'vertical', colors[filt_type])
        self._plot_single_axis(axes['axis_mediolateral'], tri_acc['mediolateral'],
                               time, 'mediolateral', colors[filt_type])
        self._plot_single_axis(axes['axis_anteroposterior'], tri_acc['anteroposterior'],
                               time, 'anteroposterior', colors[filt_type])

    def _plot_single_axis(self, ax: Axes, axis_data, time, name, color):
        ax.set_title('Axis: ' + name)
        pr = ax.plot(time, axis_data, color=color, label='Raw Data')
        handles = [pr[0]]
        title = 'Acc Data'
        ax.legend(handles=handles, title=title)

    def show_plot(self):
        plt.show()
