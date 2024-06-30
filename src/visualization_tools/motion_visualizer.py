import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class MotionVisualizer:
    def plot_acceleration_data(self, user_data: UserData):
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
        id = user_data.get_clinical_demo_data().get_id()
        fig.suptitle(f'Acceleration Data for ID: {id}', fontsize=16)
        return fig, axes

    def plot_gyroscope_data(self, user_data: UserData):
        colors = {IMUDataFilterType.RAW: 'cornflowerblue',
                  IMUDataFilterType.LPF: 'red',
                  IMUDataFilterType.KF: 'yellow'}
        fig, (ax_y, ax_p, ax_r) = self._generate_fig_axes()
        axes = {'axis_yaw': ax_y,
                'axis_pitch': ax_p,
                'axis_roll': ax_r}
        for filt_type, imu_data in user_data.get_imu_data().items():
            tri_gyr = imu_data.get_triax_gyr_data()
            time = imu_data.get_time()
            self._plot_triaxial_gyr(tri_gyr, time, filt_type, axes, colors)
        id = user_data.get_clinical_demo_data().get_id()
        fig.suptitle(f'Gyroscope Data for ID: {id}', fontsize=16)
        return fig, axes

    def _generate_fig_axes(self):
        return plt.subplots(3, sharex=True)

    def _plot_triaxial_acc(self, tri_acc, time, filt_type, axes, colors):
        axes['axis_vertical'].grid(True, 'both')
        axes['axis_mediolateral'].grid(True, 'both')
        axes['axis_anteroposterior'].grid(True, 'both')
        self._plot_single_axis(axes['axis_vertical'], tri_acc['vertical'],
                               time, 'vertical', colors[filt_type], filt_type.get_value())
        self._plot_single_axis(axes['axis_mediolateral'], tri_acc['mediolateral'],
                               time, 'mediolateral', colors[filt_type], filt_type.get_value())
        self._plot_single_axis(axes['axis_anteroposterior'], tri_acc['anteroposterior'],
                               time, 'anteroposterior', colors[filt_type], filt_type.get_value())

    def _plot_triaxial_gyr(self, tri_gyr, time, filt_type, axes, colors):
        axes['axis_yaw'].grid(True, 'both')
        axes['axis_pitch'].grid(True, 'both')
        axes['axis_roll'].grid(True, 'both')
        self._plot_single_axis(axes['axis_yaw'], tri_gyr['yaw'],
                               time, 'yaw', colors[filt_type], filt_type.get_value())
        self._plot_single_axis(axes['axis_pitch'], tri_gyr['pitch'],
                               time, 'pitch', colors[filt_type], filt_type.get_value())
        self._plot_single_axis(axes['axis_roll'], tri_gyr['roll'],
                               time, 'roll', colors[filt_type], filt_type.get_value())

    def _plot_single_axis(self, ax: Axes, axis_data, time,
                          name, color, filt_type):
        ax.set_title('Axis: ' + name)
        ax.plot(time, axis_data, color=color, label=filt_type)
        ax.legend()
        # pr = ax.plot(time, axis_data, color=color, label=filt_type)
        # handles = [pr[0]]
        # title = 'Acc Data'
        # ax.legend(handles=handles, title=title)

    def show_plot(self):
        plt.show()
