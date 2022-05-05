import os
import json
import pandas as pd
import numpy as np
import glob
import time
import binascii
import struct
from datetime import datetime

from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


DATASET_NAME = DatasetNames.FIBION


class DatasetBuilder(DatasetBuilder):
    def __init__(self, ):
        # TODO: add second sisfall dataset for the second accelerometer in dataset, currently not being used
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 0.0
        # Original units: g,g,g
        # Converted to: m/s^2,m/s^2,m/s^2
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        dataset = None
        return dataset

    def read_hex_file(self, hex_file_path):
        # Read and parse the data file
        # with open(hex_file_path, mode='r', encoding='utf8') as file:
        #     hex_file_data = file.read()
        # print(type(hex_file_data))
        #

        with open(hex_file_path, 'rb') as f:
            hexdata = f.read().hex()
        over = len(hexdata) % 24
        if over == 0:
            ixs = [i for i in range(0, len(hexdata) + 1, 24)]
            first_ix = ixs[0:-1]
            second_ix = ixs[1:]
            hexs = []
            for i, j in zip(first_ix, second_ix):
                hexs.append(hexdata[i:j])
            for hex in hexs:
                if len(hex) != 24:
                    raise ValueError('you a idiot')
                time_s, time_date = self.convert_hex_time(hex[0:12])
                x_acc = self.convert_hex_acc(hex[12:16])
                y_acc = self.convert_hex_acc(hex[16:20])
                z_acc = self.convert_hex_acc(hex[20:])


            print(hexdata[-30:-1])
        else:
            raise (ValueError('Hex file, incorrect data length'))
        # hex_list = map(''.join, zip(hexdata[::24], hexdata[1:24]))
        # print(hex_list[1:10])
        # print(hexdata[0:10000])
        # print(type(hexdata))

        # Open in binary mode (so you don 't read two byte line endings on Windows as one byte)
        # and use with statement (always do this to avoid leaked file descriptors, unflushed files)
        # with open(hex_file_path, 'rb') as f:
        #     # Slurp the whole file and efficiently convert it to hex all at once
        #     hexdata = binascii.hexlify(f.read())
        # print(hexdata[0:10000])

        # Put the acceleration data into the acceleration data classes
        # Return data classes
        return None

    def convert_hex_time(self, hex_str):
        time_s = int(hex_str, 16) / 1000
        time_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_s))
        return time_s, time_date

    def convert_hex_acc(self, hex_str):
        bits = 16  # Number of bits in a hexadecimal number format
        acc = int(hex_str, bits)
        if acc & (1 << (bits - 1)):
            acc -= 1 << bits
        acc_g = acc * 0.008
        acc_ms = acc_g * 9.8
        return acc_ms
        # acc_g = int(hex_str, 16) * 0.008
        # return acc_g

# class ConvertSisFallDataset:
#     def read_dataset(self):
#         # Initialize output
#         all_motion_data_in_dataset: List[MotionData] = []
#         # Iterate through SisFall directory for all data files
#         for root, dirs, files in list(os.walk(self.path))[1:]:
#             for file in files:
#                 if file == 'Readme.txt' or not file.endswith('.' + self.file_format):
#                     continue
#                 else:
#                     motion_file_path = os.path.join(root, file)
#                     print(motion_file_path)
#                     # Open file, read in txt file as csv data
#                     with open(motion_file_path) as mfp:
#                         data = pd.read_csv(mfp, sep=',', index_col='time')
#                         all_motion_data_in_dataset.append(
#                             self.__build_motion_data(data, file))
#         self.motion_data = all_motion_data_in_dataset
#
#     def write_dataset_to_csv(self, output_folder_path):
#         output_path = os.path.join(output_folder_path, 'SisFall_dataset_csv')
#         for motion_data in self.motion_data:
#             subject_path = os.path.join(output_path, motion_data.subject.id)
#             os.makedirs(subject_path, exist_ok=True)
#             filename = motion_data.activity.code + '_' + motion_data.subject.id + '_' + motion_data.trial + '.csv'
#             file_path = os.path.join(subject_path, filename)
#             motion_data.motion_df.to_csv(file_path, header=True)
#
#     def __apply_unit_conversion_to_all(self, data):
#         data["ADXL345_x"] = self.__apply_unit_conversion(np.array(data['ADXL345_x'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ADXL345'])
#         data["ADXL345_y"] = self.__apply_unit_conversion(np.array(data['ADXL345_y'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ADXL345'])
#         data["ADXL345_z"] = self.__apply_unit_conversion(np.array(data['ADXL345_z'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ADXL345'])
#         data["ITG3200_x"] = self.__apply_unit_conversion(np.array(data['ITG3200_x'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ITG3200'])
#         data["ITG3200_y"] = self.__apply_unit_conversion(np.array(data['ITG3200_y'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ITG3200'])
#         data["ITG3200_z"] = self.__apply_unit_conversion(np.array(data['ITG3200_z'],
#                                                                   dtype=np.float),
#                                                          self.sensor_data[
#                                                              'ITG3200'])
#         data["MMA8451Q_x"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_x'],
#                                                                    dtype=np.float),
#                                                           self.sensor_data[
#                                                               'MMA8451Q'])
#         data["MMA8451Q_y"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_y'],
#                                                                    dtype=np.float),
#                                                           self.sensor_data[
#                                                               'MMA8451Q'])
#         data["MMA8451Q_z"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_z'],
#                                                                    dtype=np.float),
#                                                           self.sensor_data[
#                                                               'MMA8451Q'])
#         return data
#
#     def __apply_unit_conversion(self, single_acc_axis, sensor: Sensor) -> np.array:
#         # Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
#         # Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
#         return np.array(single_acc_axis * ((2*sensor.range)/(2**sensor.resolution)))
#
#     def __make_time_array(self, sample_rate: int, number_of_samples: int):
#         recording_time = (number_of_samples-1)/sample_rate
#         time_array = np.linspace(0.0, recording_time, number_of_samples)
#         return time_array


def main():
    path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin\fibion_test_bin_04_10_2022.bin'
    t0 = time.time()
    db = DatasetBuilder()
    dataset = db.read_hex_file(path)
    print(str(time.time() - t0))
    print(dataset)


if __name__ == '__main__':
    main()
