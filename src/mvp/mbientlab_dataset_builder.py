import os
import csv
import numpy as np
import time
import json
from datetime import date
from dateutil import parser, tz


from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


DATASET_NAME = DatasetNames.FIBION


class MbientlabDatasetBuilder(DatasetBuilder):
    def __init__(self, timezone=tz.gettz("America/New_York")):
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 100.0
        # Original units: g,g,g
        # Converted to: m/s^2,m/s^2,m/s^2
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        self.local_tz = timezone

    def build_dataset(self, dataset_path, demo_data, clinical_demo_path,
                      segment_dataset=True, epoch_size=60.0):
        # Build user data objects
        dataset_user_data = self.build_single_user(dataset_path, {
            "user_name": "Grainger Sasso",
            "user_ID": "abcd_efgh_1234_5678",
            "age": 25,
            "height": 1.88,
            "weight": 83.91,
            "sex": "M"

        })
        # TODO: Set demographic data
        # Set dataset file
        dataset_name = 'Mbientlab'
        return Dataset(dataset_name, dataset_path, clinical_demo_path, dataset_user_data, {})

    def build_single_user(self, data_path, demo_data, demo_path):
        dataset_user_data = []
        x_data, y_data, z_data, time = self.read_mbient_file(data_path)
        imu_data_file_path = data_path
        imu_data_file_name = None
        imu_metadata_file_path = None
        units = {'vertical-acc': 'g', 'mediolateral-acc': 'g',
                 'anteroposterior-acc': 'g'}
        imu_metadata = IMUMetadata(None, self.sampling_frequency, units)
        imu_data = IMUData('', '',
                           np.array(x_data), np.array(y_data), np.array(z_data),
                           np.array([]), np.array([]), np.array([]),
                           time)
        age = self.convert_dob_age(demo_data)
        height_cm = self.convert_height_cm(demo_data)
        weight_kg = self.convert_weight_kg(demo_data)
        clinical_demo_data = ClinicalDemographicData(
            demo_data['user_ID'], age, demo_data['gender'],
            False, height_cm, None,
            demo_data['user_name'], weight_kg,
        )
        user_data = [UserData(
            imu_data_file_path, imu_data_file_name, imu_metadata_file_path,
            demo_path, {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data
        )]
        return user_data

    def convert_weight_kg(self, demo_data):
        weight_lbs = int(demo_data['weight_lbs'])
        return weight_lbs * 0.4536

    def convert_height_cm(self, demo_data):
        height_ft_in = demo_data['height_ft_in']
        h_ft = int(height_ft_in[0:2])
        h_in = int(height_ft_in[3:])
        add_in = h_ft * 12
        total_h_in = h_in + add_in
        return total_h_in * 2.54

    def convert_dob_age(self, demo_data):
        dob = demo_data['DOB_mm_dd_yyyy']
        year = int(dob[6:])
        month = int(dob[0:2])
        day = int(dob[3:5])
        dob_date = date(year, month, day)
        today = date.today()
        return float(today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day)))

    def load_clinical_demo_data(self, path):
        with open(path, 'r') as f:
            data = json.loads(f)
            f.close()
        return data

    def set_demo_data(self, demo_data):
        user_name = demo_data['user_name']
        user_id = demo_data['user_ID']
        age = demo_data['age']
        height = demo_data['height']
        weight = demo_data['weight']
        sex = demo_data['sex']
        demo = ClinicalDemographicData(
            user_id, age, sex, False, height, None, user_name, weight
        )
        return demo

    def read_mbient_file(self, path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            x_data = []
            y_data = []
            z_data = []
            time = []
            row_keys = reader.fieldnames
            relevant_keys = self.get_relevant_keys(row_keys)
            for row in reader:
                x_data.append(float(row[relevant_keys['x']]))
                y_data.append(float(row[relevant_keys['y']]))
                z_data.append(float(row[relevant_keys['z']]))
                if len(str(row[relevant_keys['time']])) > 1650000000000:
                    time.append(float(row[relevant_keys['time']]) / 1000.0)
                else:
                    time.append(float(row[relevant_keys['time']]))
            f.close()
        # x_data = x_data[2383971:]
        # y_data = y_data[2383971:]
        # z_data = z_data[2383971:]
        # time = time[2383971:]
        x_data = np.array(x_data)
        x_data = np.float16(x_data)
        y_data = np.array(y_data)
        y_data = np.float16(y_data)
        z_data = np.array(z_data)
        z_data = np.float16(z_data)
        time = np.array(time)
        if
        return x_data, y_data, z_data, time

    def get_relevant_keys(self, row_keys):
        relevant_keys = {}
        if ('x-axis (g)' in row_keys) and ('y-axis (g)' in row_keys) and ('z-axis (g)' in row_keys) and (
                'epoch (ms)' in row_keys):
            relevant_keys['x'] = 'x-axis (g)'
            relevant_keys['y'] = 'y-axis (g)'
            relevant_keys['z'] = 'z-axis (g)'
            relevant_keys['time'] = 'epoch (ms)'
        elif ('x-axis (g)' in row_keys) and ('y-axis (g)' in row_keys) and ('z-axis (g)' in row_keys) and (
                'epoc (ms)' in row_keys):
            relevant_keys['x'] = 'x-axis (g)'
            relevant_keys['y'] = 'y-axis (g)'
            relevant_keys['z'] = 'z-axis (g)'
            relevant_keys['time'] = 'epoc (ms)'
        elif ('x' in row_keys) and ('y' in row_keys) and ('z' in row_keys) and ('epoch' in row_keys):
            relevant_keys['x'] = 'x'
            relevant_keys['y'] = 'y'
            relevant_keys['z'] = 'z'
            relevant_keys['time'] = 'epoch'
        elif ('X' in row_keys) and ('Y' in row_keys) and ('Z' in row_keys) and ('Epoch' in row_keys):
            relevant_keys['x'] = 'X'
            relevant_keys['y'] = 'Y'
            relevant_keys['z'] = 'Z'
            relevant_keys['time'] = 'Epoch'
        else:
            raise ValueError(f'Invalid access keys for mbient file columns: {row_keys}')
        return relevant_keys

def main():
    path = r'/home/grainger/Desktop/datasets/mbientlab/test/MULTIDAY_MetaWear_2022-08-19T12.38.00.909_C85D72EF7FA2_Accelerometer.csv'
    t0 = time.time()
    db = MbientlabDatasetBuilder()
    dataset = db.build_dataset(path, '', '')
    print(f'Time: {str(time.time() - t0)}')
    print(dataset)


if __name__ == '__main__':
    main()
