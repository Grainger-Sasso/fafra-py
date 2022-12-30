import os
import matplotlib
import pandas as pd
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

    def trim_data(self, data_path, start, end, output_path, viz=False):
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
        # Trim the data given the specified times
        st_end_ixs = self.get_epoch_ixs(start, end, time)
        header = ['epoch', 'x', 'y', 'z']
        x_data = x_data[st_end_ixs[0]: st_end_ixs[1]]
        y_data = y_data[st_end_ixs[0]: st_end_ixs[1]]
        z_data = z_data[st_end_ixs[0]: st_end_ixs[1]]
        time = time[st_end_ixs[0]: st_end_ixs[1]]
        # Format data for output to CSV
        output_df = pd.DataFrame(
            {
                'epoch': time,
                'x': x_data,
                'y': y_data,
                'z': z_data
            }
        )
        # Write data to specified file path
        file_name = 'TRIMMED_' + os.path.basename(data_path)
        output_path = os.path.join(output_path, file_name)
        output_df.to_csv(output_path, header=True, index=False)
        print('correct')

        pass

    def get_epoch_ixs(self, start, end, time):
        ixs = []
        start_found = False
        end_found = False
        for ix, epoch in enumerate(time):
            if epoch >= start and not start_found:
                ixs.append(ix)
                start_found = True
            if epoch >= end and not end_found:
                ixs.append(ix)
                end_found = True
            if start_found and end_found:
                break
        return ixs


def main():
    trimmer = MbientDataTrimmer()
    data_path = '/home/grainger/Desktop/skdh_testing/mbient_usb_data/Bridges_12_2022/TRIMMED_imu_data_device_003_12-12-2022-11-45-17.csv'
    #'TRIMMED_imu_data_device_003_12-12-2022-11-45-17'
    # 'TRIMMED_imu_data_device_004_12-12-2022-18-43-20'
    # Epoch corresponding to 2022_12_08 at 6:00 PM local time
    start = 1670540400
    # Epoch corresponding to 2022_12_10 at 5:00 PM local time
    end = 1670709600
    output_path = '/home/grainger/Desktop/skdh_testing/mbient_usb_data/Bridges_12_2022/'
    trimmer.trim_data(data_path, start, end, output_path, viz=True)


if __name__ == '__main__':
    main()
