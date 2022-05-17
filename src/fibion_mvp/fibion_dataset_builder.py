import os
import json
import pandas as pd
import numpy as np
import glob
import time
import binascii
import struct
from matplotlib import pyplot as plt
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
        dataset_user_data = []
        hex_file_paths = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        for hex_file_path in hex_file_paths:
            user_data = self.read_hex_file(hex_file_path)
            dataset_user_data.append(user_data)

        return dataset

    def read_hex_file(self, hex_file_path):
        with open(hex_file_path, 'rb') as f:
            hexdata = f.read().hex()
        if len(hexdata) % 24 == 0:
            ixs = [i for i in range(0, len(hexdata) + 1, 24)]
            first_ix = ixs[0:-1]
            second_ix = ixs[1:]
            hexs = []
            t0 = time.time()
            for i, j in zip(first_ix, second_ix):
                hexs.append(hexdata[i:j])
            t1 = time.time()
            print(t1-t0)
            time_s = []
            time_date = []
            x_acc = []
            y_acc = []
            z_acc = []
            for hex in hexs:
                if len(hex) != 24:
                    raise ValueError(f'Hex bin data length not 24: {len(hexs)}')
                time_s_i, time_date_i = self.convert_hex_time(hex[0:12])
                time_s.append(time_s_i)
                time_date.append(time_date_i)
                x_acc.append(self.convert_hex_acc(hex[12:16]))
                y_acc.append(self.convert_hex_acc(hex[16:20]))
                z_acc.append(self.convert_hex_acc(hex[20:]))
            # TODO: set the axes to correct anatomical orientation
            t2 = time.time()
            print(t2 - t1)
            imu_data = IMUData('', '',
                               np.array(y_acc), np.array(x_acc), np.array(z_acc),
                               np.array([]), np.array([]), np.array([]),
                               time_s)
        else:
            raise (ValueError('Hex file, incorrect data length'))
        imu_data_file_path = hex_file_path
        imu_data_file_name = os.path.basename(hex_file_path)
        imu_metadata_file_path = None
        clinical_demo_file_path = None
        units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                 'anteroposterior-acc': 'm/s^2'}
        # TODO: Use ID and hashmap system to map clinical demographic data to users
        clinical_demo_data = ClinicalDemographicData('', 0.0, '', True, 0.0,
                                                     None)
        # TODO: get the correct sampling frequency
        imu_metadata = IMUMetadata(None, 24.0, units)
        user_data = UserData(
            imu_data_file_path, imu_data_file_name, imu_metadata_file_path,
            clinical_demo_file_path, imu_data, imu_metadata, clinical_demo_data
        )
        return user_data

    def convert_hex_time(self, hex_str):
        t0 = time.time()
        time_s = int(hex_str, 16) / 1000
        time_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_s))
        print(f'time_conversion: {time.time()-t0}')
        return time_s, time_date

    def convert_hex_acc(self, hex_str):
        t0 = time.time()
        bits = 16  # Number of bits in a hexadecimal number format
        acc = int.from_bytes(bytearray.fromhex(hex_str), byteorder='big', signed=True)
        acc_g = acc * 0.008
        acc_ms = acc_g * 9.8
        print(f'acc_conversion: {time.time() - t0}')
        return acc_ms
        # acc_g = int(hex_str, 16) * 0.008
        # return acc_g


def main():
    path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin\fibion_test_bin_04_10_2022.bin'
    t0 = time.time()
    db = DatasetBuilder()
    dataset = db.read_hex_file(path)
    print(f'Time: {str(time.time() - t0)}')
    print(dataset)


if __name__ == '__main__':
    main()
