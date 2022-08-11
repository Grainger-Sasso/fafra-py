import os
import json
import pandas as pd
import numpy as np
import time
from dateutil import parser, tz
import glob
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


class FibionDatasetBuilder(DatasetBuilder):
    def __init__(self, timezone=tz.gettz("America/New_York")):
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 25.0
        # Original units: g,g,g
        # Converted to: m/s^2,m/s^2,m/s^2
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        self.local_tz = timezone

    def build_dataset(self, dataset_path, demo_data, clinical_demo_path,
                      segment_dataset=True, epoch_size=60.0):
        dataset = None
        dataset_user_data = []
        hex_file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        for hex_file_path in hex_file_paths:
            user_data = self.read_hex_file(hex_file_path, demo_data, segment_dataset, epoch_size)
            dataset_user_data.extend(user_data)
        dataset_name = 'Fibion'
        return Dataset(dataset_name, dataset_path, clinical_demo_path, dataset_user_data, {})

    def build_single_user(self, data_path, demo_data):
        dataset = None
        dataset_user_data = []
        user_data = self.read_hex_file(data_path, demo_data, False, 0.0)
        dataset_user_data.extend(user_data)
        dataset_name = 'Fibion'
        return Dataset(dataset_name, data_path, '', dataset_user_data, {})

    def read_hex_file(self, hex_file_path, demo_data, segment_dataset, epoch_size):
        with open(hex_file_path, 'rb') as f:
            hexdata = f.read().hex()
        if len(hexdata) % 24 == 0:
            ixs = [i for i in range(0, len(hexdata) + 1, 24)]
            first_ix = ixs[0:-1]
            second_ix = ixs[1:]
            hexs = []
            for i, j in zip(first_ix, second_ix):
                hexs.append(hexdata[i:j])
            time_s = []
            x_acc = []
            y_acc = []
            z_acc = []
            for hex in hexs:
                if len(hex) != 24:
                    raise ValueError(f'Hex bin data length not 24: {len(hexs)}')
                time_s_i = self.convert_hex_time(hex[0:12])
                time_s.append(time_s_i)
                x_acc.append(self.convert_hex_acc(hex[12:16]))
                y_acc.append(self.convert_hex_acc(hex[16:20]))
                z_acc.append(self.convert_hex_acc(hex[20:]))
            # TODO: set the axes to correct anatomical orientation
        else:
            raise (ValueError('Hex file, incorrect data length'))
        imu_data_file_path = hex_file_path
        imu_data_file_name = os.path.basename(hex_file_path)
        imu_metadata_file_path = None
        clinical_demo_file_path = None
        units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                 'anteroposterior-acc': 'm/s^2'}
        # TODO: Use ID and hashmap system to map clinical demographic data to users
        clinical_demo_data = ClinicalDemographicData('', 0.0, '', False, demo_data['user_height'],
                                                     None)
        # TODO: get the correct sampling frequency
        imu_metadata = IMUMetadata(None, self.sampling_frequency, units)
        if segment_dataset:
            # Create many user data objects, return them all
            data = np.array([time_s, x_acc, y_acc, z_acc])
            data = data.T
            data_segments = self.segment_data(data, epoch_size, self.sampling_frequency)
            user_data = []
            for segment in data_segments:
                time_s = segment.T[0]
                x_acc = segment.T[1]
                y_acc = segment.T[2]
                z_acc = segment.T[3]
                imu_data = IMUData('', '',
                                   np.array(x_acc), np.array(z_acc),
                                   np.array(y_acc),
                                   np.array([]), np.array([]), np.array([]),
                                   time_s)
                user_data.append(UserData(
                imu_data_file_path, imu_data_file_name, imu_metadata_file_path,
                clinical_demo_file_path, {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data
            ))
        else:
            imu_data = IMUData('', '',
                               np.array(y_acc), np.array(x_acc), np.array(z_acc),
                               np.array([]), np.array([]), np.array([]),
                               time_s)
            user_data = [UserData(
                imu_data_file_path, imu_data_file_name, imu_metadata_file_path,
                clinical_demo_file_path, {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data
            )]
        return user_data

    def convert_hex_time(self, hex_str):
        # Converts from hex to s (in CEST time, Stockholm, Sweden)
        swe_time_s = (int(hex_str, 16) / 1000)
        # TODO: enable proper tz conversion; disabled now because of runtime
        # local_time_s = self.convert_tz(swe_time_s)
        # local_time_s = swe_time_s - (3600 * 6)
        local_time_s =swe_time_s
        return local_time_s

    def convert_tz(self, from_s_swe):
        swe_time_date = time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(from_s_swe))
        swe_time_date = parser.parse(swe_time_date).replace(
            tzinfo=tz.gettz('Europe/Stockholm'))
        # Convert UTC datetime to Local datetime
        local_time_date = swe_time_date.astimezone(self.local_tz)
        # Convert local datetime to epoch
        local_time_s = local_time_date.timestamp()
        return local_time_s

    def convert_hex_acc(self, hex_str):
        bits = 16  # Number of bits in a hexadecimal number format
        acc = int.from_bytes(bytearray.fromhex(hex_str), byteorder='big', signed=True)
        acc_g = acc * 0.008
        acc_ms = acc_g * 9.8
        return acc_ms
        # acc_g = int(hex_str, 16) * 0.008
        # return acc_g


def main():
    path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin'
    t0 = time.time()
    db = FibionDatasetBuilder()
    dataset = db.build_dataset(path, '')
    print(f'Time: {str(time.time() - t0)}')
    print(dataset)


if __name__ == '__main__':
    main()
