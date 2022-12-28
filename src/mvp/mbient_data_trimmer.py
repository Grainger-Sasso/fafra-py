import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdate
from src.mvp.mbientlab_dataset_builder import MbientlabDatasetBuilder


class MbientDataTrimmer:
    def __init__(self):
        self.data_keys = {
            'x': 'x-axis (g)',
            'y': 'y-axis (g)',
            'z': 'z-axis (g)',
            'time': 'epoch (ms)'
        }

    def trim_data(self, data_path, start, end, ouput_path, viz=False):
        # read in the data to be trimmed
        dsb = MbientlabDatasetBuilder()
        x_data, y_data, z_data, time = dsb.read_mbient_file(data_path)
        # Convert epoch time to indexes
        # visualize if specified, include the times to trim
        if viz:
            matplotlib.rcParams['timezone'] = 'US/Eastern'
            secs = mdate.epoch2num(time)
            fig, ax = plt.subplots()
            ax.plot_date(secs, y_data, linestyle='solid', marker='None')
            # ADDS VERTICAL LINES AT BOUT IXS DETECTED
            # flat_ixs = [ix for sub_ix in bout_ixs for ix in sub_ix]
            # flat_ixs = list(set(flat_ixs))
            # for ix in flat_ixs:
            #     plt.axvline(x=secs[ix], color='r')
            #####
            date_fmt = '%d-%m-%y %H:%M:%S'
            date_formatter = mdate.DateFormatter(date_fmt)
            ax.xaxis.set_major_formatter(date_formatter)
            fig.autofmt_xdate()
            plt.show()
        # trim the data given the specified times
        # write file to specified folder
        pass


def main():
    trimmer = MbientDataTrimmer()
    data_path = '/home/grainger/Desktop/skdh_testing/mbient_usb_data/Bridges_12_2022/imu_data_device_002_12-11-2022-12-32-45.csv'
    # Epoch corresponding to 2022_12_08 at 6:00 PM local time
    start = 1670540400
    end = 1670698800
    output_path = '/home/grainger/Desktop/skdh_testing/mbient_usb_data/Bridges_12_2022/'
    trimmer.trim_data(data_path, start, end, output_path)


if __name__ == '__main__':
    main()
