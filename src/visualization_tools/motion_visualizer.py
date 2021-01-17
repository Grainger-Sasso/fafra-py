import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List
from src.dataset_tools.motion_data.acceleration.triaxial_acceleration import TriaxialAcceleration
from src.dataset_tools.motion_data.acceleration.acceleration import Acceleration
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.dataset_tools.params.subject import Subject
from src.dataset_tools.params.activity import Activity


class MotionVisualizer:

    def plot_motion_data(self, dataset: MotionDataset, subject, activity, trial):
        motion_data = dataset.get_data(subject, activity, trial)
        subject_data = motion_data.get_subject()
        activity_data = motion_data.get_activity()
        self.__plot_triaxial_acc(motion_data.tri_lin_accs, subject_data, activity_data, trial)
        self.__plot_triaxial_acc(motion_data.tri_ang_accs, subject_data, activity_data, trial)

    def __plot_triaxial_acc(self, tri_accs: List[TriaxialAcceleration], subject_data: Subject, activity_data: Activity,
                            trial):
        raw_color = 'navajowhite'
        lp_filtered_color = 'cornflowerblue'
        kf_filtered_color = 'deeppink'
        first_derivative_color = 'olivedrab'
        colors = {'raw': raw_color, 'lp_filtered': lp_filtered_color, 'kf_filtered': kf_filtered_color, 'first_derivative': first_derivative_color}
        if tri_accs:
            for tri_acc in tri_accs:
                fig, (ax_x, ax_y, ax_z) = plt.subplots(3, sharex=True)
                fig.suptitle(tri_acc.name + ': ' + tri_acc.sensor_name + '\n Subject Data: ' + subject_data.get_subject_identifier() + ', ' + str(subject_data.get_subject_age()) + ', ' + subject_data.get_subject_gender() + '\n Activity: ' + activity_data.get_description())
                # Plot x axis data
                self.__plot_single_axis(ax_x, tri_acc.x_axis, colors)
                self.__plot_single_axis(ax_y, tri_acc.y_axis, colors)
                self.__plot_single_axis(ax_z, tri_acc.z_axis, colors)
            plt.show()

    def __plot_single_axis(self, ax: Axes, axis_data: Acceleration, colors):
        ax.set_title('Axis: ' + axis_data.axis + ' - ' + axis_data.anatomical_axis)
        pr = ax.plot(axis_data.time, axis_data.acceleration_data, color=colors['raw'], label='Raw Data')
        handles = [pr[0]]
        title = 'Raw Data'
        if axis_data.kf_filtered_data.any():
            pkf = ax.plot(axis_data.time, axis_data.kf_filtered_data, color=colors['kf_filtered'], label='KF Filtered Data')
            handles.append(pkf[0])
            title += ', Kalman Filtered Data'
        if axis_data.lp_filtered_data.any():
            plpf = ax.plot(axis_data.time, axis_data.lp_filtered_data, color=colors['lp_filtered'], label='LP Filtered Data')
            handles.append(plpf[0])
            title += ', Low-pass Filtered Data'

        # TODO: Put first derivative on seperate graph, scale and layout does not work with the other filtered data + it's sort of unrelated to raw and filtered data
        # if axis_data.first_derivative_data.any():
        #     pfd = ax.plot(axis_data.time[1:], axis_data.first_derivative_data, color=colors['first_derivative'], label='First Derivative (dx/dt) Data')
        #     handles.append(pfd[0])
        #     title += ', and First Derivative (dx/dt) Data'

        ax.legend(handles=handles, title=title)

