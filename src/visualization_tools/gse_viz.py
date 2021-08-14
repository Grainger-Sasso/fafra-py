import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.datasets.ltmm.ltmm_dataset import LTMMData


class GSEViz:

    def plot_motion_data(self, v_acc, ml_acc, ap_acc, sampling_frequency):
        tri_lin_acc = [v_acc, ml_acc, ap_acc]
        sampling_freq = ltmm_data.get_sampling_frequency()
        self.__plot_triaxial_acc(tri_lin_acc, sampling_freq)

    def __plot_triaxial_acc(self, tri_acc, sampling_freq):
        raw_color = 'cornflowerblue'
        colors = {'raw': raw_color}
        fig, (ax_x, ax_y, ax_z) = plt.subplots(3, sharex=True)
        ax_x.grid(True, 'both')
        ax_y.grid(True, 'both')
        ax_z.grid(True, 'both')
        time = np.linspace(0, len(tri_acc[0])/int(sampling_freq), len(tri_acc[0]))
        self.__plot_single_axis(ax_x, tri_acc[0], time, 'vertical', colors)
        self.__plot_single_axis(ax_y, tri_acc[1], time, 'mediolateral', colors)
        self.__plot_single_axis(ax_z, tri_acc[2], time, 'anteroposterior', colors)
        # self.show_plot()

    def __plot_single_axis(self, ax: Axes, axis_data, time, name, colors):
        ax.set_title('Axis: ' + name)
        pr = ax.plot(time, axis_data, color=colors['raw'], label='Raw Data')
        handles = [pr[0]]
        title = 'Raw Data'
        ax.legend(handles=handles, title=title)

    def show_plot(self):
        plt.show()

def main():
    print('yup')
    viz = GSEViz()
    viz.plot_motion_data(ltmm_data)
    print('uh huh')

if __name__ == '__main__':
    main()
