import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.datasets.ltmm.ltmm_dataset import LTMMData


class GSEViz:

    def plot_motion_data(self, ltmm_data: LTMMData):
        tri_lin_acc = [ltmm_data.get_axis_acc_data('vertical'),
                       ltmm_data.get_axis_acc_data('mediolateral'),
                       ltmm_data.get_axis_acc_data('anteroposterior'),]
        sampling_freq = ltmm_data.get_sampling_frequency()
        self.__plot_triaxial_acc(tri_lin_acc, sampling_freq)

    def __plot_triaxial_acc(self, tri_acc, sampling_freq):
        raw_color = 'cornflowerblue'
        colors = {'raw': raw_color}
        fig, (ax_x, ax_y, ax_z) = plt.subplots(3, sharex=True)
        time = np.linspace(0, len(tri_acc[0]), sampling_freq)
        self.__plot_single_axis(ax_x, tri_acc[0], time, 'vertical', colors)
        self.__plot_single_axis(ax_y, tri_acc[1], time, 'mediolateral', colors)
        self.__plot_single_axis(ax_z, tri_acc[2], time, 'anteroposterior', colors)
        plt.show()

    def __plot_single_axis(self, ax: Axes, axis_data, time, name, colors):
        ax.set_title('Axis: ' + name)
        pr = ax.plot(time, axis_data.acceleration_data, color=colors['raw'], label='Raw Data')
        handles = [pr[0]]
        title = 'Raw Data'
        ax.legend(handles=handles, title=title)

def main():
    print('yup')
    viz = GSEViz()
    viz.plot_motion_data(ltmm_data)
    print('uh huh')

if __name__ == '__main__':
    main()
