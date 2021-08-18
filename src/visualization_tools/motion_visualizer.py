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
        fig, (ax_x, ax_y, ax_z) = self._generate_fig_axes()
        axes = [ax_x, ax_y, ax_z]
        for filt_type, imu_data in user_data.get_imu_data().items():
            tri_lin_acc = imu_data.get_triax_acc_data()
            sampling_freq = user_data.get_imu_metadata().get_sampling_frequency()
            self._plot_triaxial_acc(tri_lin_acc, sampling_freq,
                                    filt_type, axes, colors)
        return fig, axes

    def _generate_fig_axes(self):
        return plt.subplots(3, sharex=True)

    def _plot_triaxial_acc(self, tri_acc, sampling_freq,
                           filt_type, axes, colors):
        axes[0].grid(True, 'both')
        axes[1].grid(True, 'both')
        axes[2].grid(True, 'both')
        time = np.linspace(0, len(tri_acc['vertical']) / int(sampling_freq),
                           len(tri_acc['vertical']))
        self._plot_single_axis(axes[0], tri_acc['vertical'],
                               time, 'vertical', colors[filt_type])
        self._plot_single_axis(axes[1], tri_acc['mediolateral'],
                               time, 'mediolateral', colors[filt_type])
        self._plot_single_axis(axes[2], tri_acc['anteroposterior'],
                               time, 'anteroposterior', colors[filt_type])

    def _plot_single_axis(self, ax: Axes, axis_data, time, name, color):
        ax.set_title('Axis: ' + name)
        pr = ax.plot(time, axis_data, color=color, label='Raw Data')
        handles = [pr[0]]
        title = 'Acc Data'
        ax.legend(handles=handles, title=title)

    def show_plot(self):
        plt.show()
