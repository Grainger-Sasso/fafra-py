import os
import pandas as pd
import numpy as np
import glob
from typing import List

from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


DATASET_NAME = DatasetNames.SISFALL


class DatasetBuilder(DatasetBuilder):
    def __init__(self, ):
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 200.0
        self.units = {'vertical-acc': 'g', 'mediolateral-acc': 'g',
                      'anteroposterior-acc': 'g',
                      'yaw': 'deg/s', 'pitch': 'deg/s', 'roll': 'deg/s'}

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        self._generate_data_file_paths(dataset_path)
        dataset = []
        # for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
        #     data_file_path = header_and_data_file_path['data_file_path']
        #     header_file_path = header_and_data_file_path['header_file_path']
        #     data_path = os.path.splitext(data_file_path)[0]
        #     header_path = os.path.splitext(header_file_path)[0]
        #     wfdb_record = wfdb.rdrecord(data_path)
        #     id = wfdb_record.record_name
        #     data = np.array(wfdb_record.p_signal)
        #     data = np.float16(data)
        #     # Convert acceleration data from g to m/s^2
        #     data[:, 0:3] = data[:, 0:3] * 9.81
        #     header_data = wfdb.rdheader(header_path)
        #     if wfdb_record.comments[0][4:]:
        #         age = float(wfdb_record.comments[0][4:])
        #     sex = wfdb_record.comments[1][4:]
        #     if id.casefold()[0] == 'f':
        #         faller_status = True
        #     elif id.casefold()[0] == 'c':
        #         faller_status = False
        #     else:
        #         raise ValueError('LTMM Data faller status unclear from id')
        #
        #     imu_data_file_path: str = data_file_path
        #     imu_metadata_file_path: str = header_file_path
        #     clinical_demo_file_path: str = 'N/A'
        #     imu_metadata = IMUMetadata(header_data, self.sampling_frequency, self.units)
        #     clinical_demo_data = ClinicalDemographicData(id, age, sex, faller_status, self.height)
        #     if segment_dataset:
        #         #TODO: track the segmented data with a linked list
        #         # Segment the data and build a UserData object for each epoch
        #         data_segments = self.segment_data(data, epoch_size, self.sampling_frequency)
        #         for segment in data_segments:
        #             imu_data = self._generate_imu_data_instance(segment, self.sampling_frequency)
        #             dataset.append(UserData(imu_data_file_path, imu_metadata_file_path, clinical_demo_file_path,
        #                                     {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        #     else:
        #         # Build a UserData object for the whole data
        #         imu_data = self._generate_imu_data_instance(data, self.sampling_frequency)
        #         dataset.append(UserData(imu_data_file_path, imu_metadata_file_path, clinical_demo_file_path,
        #                                 {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        # return Dataset(self.get_dataset_name(), dataset_path, clinical_demo_path, dataset)

    def generate_data_file_paths(self, dataset_path):
        data_file_paths = {}
        # Iterate through all of the files in the CSV directory, get all filenames
        for folder_name in next(os.walk(dataset_path))[1]:
            folder_path = os.path.join(dataset_path, folder_name)
            for data_file_path in glob.glob(
                os.path.join(folder_path, '*.csv')):
                print(data_file_path)
        # paths = glob.glob(os.path.join(dataset_path, '/*'))
        # paths = glob.glob(dataset_path)
        # Match the file paths to participant metadata
        # Get all data file paths
        for data_file_path in glob.glob(os.path.join(dataset_path, '*.csv')):
            data_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
            data_file_paths[data_file_name] = data_file_path

        # Match corresponding data and header files
        for name, path in data_file_paths.items():
            pass

    # def read_dataset(self):
    #     # Initialize output
    #     all_motion_data_in_dataset: List[MotionData] = []
    #     # Iterate through SisFall directory for all data files
    #     for root, dirs, files in list(os.walk(self.path))[1:]:
    #         for file in files:
    #             if file == 'Readme.txt' or not file.endswith('.' + self.file_format):
    #                 continue
    #             else:
    #                 motion_file_path = os.path.join(root, file)
    #                 print(motion_file_path)
    #                 # Open file, read in txt file as csv data
    #                 with open(motion_file_path) as mfp:
    #                     data = pd.read_csv(mfp, sep=',', index_col='time')
    #                     all_motion_data_in_dataset.append(
    #                         self.__build_motion_data(data, file))
    #     self.motion_data = all_motion_data_in_dataset
    #
    # def write_dataset_to_csv(self, output_folder_path):
    #     output_path = os.path.join(output_folder_path, 'SisFall_dataset_csv')
    #     for motion_data in self.motion_data:
    #         print(motion_data.subject.id)
    #         subject_path = os.path.join(output_path, motion_data.subject.id)
    #         os.makedirs(subject_path, exist_ok=True)
    #         filename = motion_data.activity.code + '_' + motion_data.subject.id + '_' + motion_data.trial + '.csv'
    #         file_path = os.path.join(subject_path, filename)
    #         motion_data.motion_df.to_csv(file_path, header=True, index=False)
    #
    # def __build_motion_data(self, data, file):
    #     # Build out acceleration instances from data
    #     adxl_x = LinearAcceleration('X', 'Frontal', np.array(data['ADXL345_x']), np.array(data.index), self.sensor_data['ADXL345'])
    #     adxl_y = LinearAcceleration('Y', 'Vertical', np.array(data['ADXL345_y']), np.array(data.index), self.sensor_data['ADXL345'])
    #     adxl_z = LinearAcceleration('Z', 'Sagittal', np.array(data['ADXL345_z']), np.array(data.index), self.sensor_data['ADXL345'])
    #     adxl_tri_acc = TriaxialLinearAcceleration('Triaxial Linear', adxl_x, adxl_y, adxl_z)
    #     # Build out acceleration instances from data
    #     itg_x = AngularAcceleration('X', 'Frontal', np.array(data['ITG3200_x']), np.array(data.index), self.sensor_data['ITG3200'])
    #     itg_y = AngularAcceleration('Y', 'Vertical', np.array(data['ITG3200_y']), np.array(data.index), self.sensor_data['ITG3200'])
    #     itg_z = AngularAcceleration('Z', 'Sagittal', np.array(data['ITG3200_z']), np.array(data.index), self.sensor_data['ITG3200'])
    #     itg_tri_acc = TriaxialAngularAcceleration('Triaxial Angular', itg_x, itg_y, itg_z)
    #     # Build out acceleration instances from data
    #     mma_x = LinearAcceleration('X', 'Frontal', np.array(data['MMA8451Q_x']), np.array(data.index), self.sensor_data['MMA8451Q'])
    #     mma_y = LinearAcceleration('Y', 'Vertical', np.array(data['MMA8451Q_y']), np.array(data.index), self.sensor_data['MMA8451Q'])
    #     mma_z = LinearAcceleration('Z', 'Sagittal', np.array(data['MMA8451Q_z']), np.array(data.index), self.sensor_data['MMA8451Q'])
    #     mma_tri_acc = TriaxialLinearAcceleration('Triaxial Linear', mma_x, mma_y, mma_z)
    #     # Get activity code, subject id, and trial from file name scheme
    #     activity_code, subject_id, trial = os.path.splitext(file)[0].split('_')
    #     subject = self.__subject_id_mapper(subject_id)
    #     activity = self.__activity_code_mapper(activity_code)
    #     # Build motion data
    #     motion_data = MotionData(subject, activity, trial, data,
    #                              [adxl_tri_acc, mma_tri_acc], [itg_tri_acc])
    #     return motion_data
    #
    # def __subject_id_mapper(self, subject_id):
    #     self.subject_data = {'SA01': {'id': 'SA01', 'age': 26, 'height': 165, 'weight': 53, 'gender': 'F'},
    #               'SA02': {'id': 'SA02', 'age': 23, 'height': 176, 'weight': 58.5, 'gender': 'M'},
    #               'SA03': {'id': 'SA03', 'age': 19, 'height': 156, 'weight': 48, 'gender': 'F'},
    #               'SA04': {'id': 'SA04', 'age': 23, 'height': 170, 'weight': 72, 'gender': 'M'},
    #               'SA05': {'id': 'SA05', 'age': 22, 'height': 172, 'weight': 69.5, 'gender': 'M'},
    #               'SA06': {'id': 'SA06', 'age': 21, 'height': 169, 'weight': 58, 'gender': 'M'},
    #               'SA07': {'id': 'SA07', 'age': 21, 'height': 156, 'weight': 63, 'gender': 'F'},
    #               'SA08': {'id': 'SA08', 'age': 21, 'height': 149, 'weight': 41.5, 'gender': 'F'},
    #               'SA09': {'id': 'SA09', 'age': 24, 'height': 165, 'weight': 64, 'gender': 'M'},
    #               'SA10': {'id': 'SA10', 'age': 21, 'height': 177, 'weight': 67, 'gender': 'M'},
    #               'SA11': {'id': 'SA11', 'age': 19, 'height': 170, 'weight': 80.5, 'gender': 'M'},
    #               'SA12': {'id': 'SA12', 'age': 25, 'height': 153, 'weight': 47, 'gender': 'F'},
    #               'SA13': {'id': 'SA13', 'age': 22, 'height': 157, 'weight': 55, 'gender': 'F'},
    #               'SA14': {'id': 'SA14', 'age': 27, 'height': 160, 'weight': 46, 'gender': 'F'},
    #               'SA15': {'id': 'SA15', 'age': 25, 'height': 160, 'weight': 52, 'gender': 'F'},
    #               'SA16': {'id': 'SA16', 'age': 20, 'height': 169, 'weight': 61, 'gender': 'F'},
    #               'SA17': {'id': 'SA17', 'age': 23, 'height': 182, 'weight': 75, 'gender': 'M'},
    #               'SA18': {'id': 'SA18', 'age': 23, 'height': 181, 'weight': 73, 'gender': 'M'},
    #               'SA19': {'id': 'SA19', 'age': 30, 'height': 170, 'weight': 76, 'gender': 'M'},
    #               'SA20': {'id': 'SA20', 'age': 30, 'height': 150, 'weight': 42, 'gender': 'F'},
    #               'SA21': {'id': 'SA21', 'age': 30, 'height': 183, 'weight': 68, 'gender': 'M'},
    #               'SA22': {'id': 'SA22', 'age': 19, 'height': 158, 'weight': 50.5, 'gender': 'F'},
    #               'SA23': {'id': 'SA23', 'age': 24, 'height': 156, 'weight': 48, 'gender': 'F'},
    #               'SE01': {'id': 'SE01', 'age': 71, 'height': 171, 'weight': 102, 'gender': 'M'},
    #               'SE02': {'id': 'SE02', 'age': 75, 'height': 150, 'weight': 57, 'gender': 'F'},
    #               'SE03': {'id': 'SE03', 'age': 62, 'height': 150, 'weight': 51, 'gender': 'F'},
    #               'SE04': {'id': 'SE04', 'age': 63, 'height': 160, 'weight': 59, 'gender': 'F'},
    #               'SE05': {'id': 'SE05', 'age': 63, 'height': 165, 'weight': 72, 'gender': 'M'},
    #               'SE06': {'id': 'SE06', 'age': 60, 'height': 163, 'weight': 79, 'gender': 'M'},
    #               'SE07': {'id': 'SE07', 'age': 65, 'height': 168, 'weight': 76, 'gender': 'M'},
    #               'SE08': {'id': 'SE08', 'age': 68, 'height': 163, 'weight': 72, 'gender': 'F'},
    #               'SE09': {'id': 'SE09', 'age': 66, 'height': 167, 'weight': 65, 'gender': 'M'},
    #               'SE10': {'id': 'SE10', 'age': 64, 'height': 156, 'weight': 66, 'gender': 'F'},
    #               'SE11': {'id': 'SE11', 'age': 66, 'height': 169, 'weight': 63, 'gender': 'F'},
    #               'SE12': {'id': 'SE12', 'age': 69, 'height': 164, 'weight': 56.5, 'gender': 'M'},
    #               'SE13': {'id': 'SE13', 'age': 65, 'height': 171, 'weight': 72.5, 'gender': 'M'},
    #               'SE14': {'id': 'SE14', 'age': 67, 'height': 163, 'weight': 58, 'gender': 'M'},
    #               'SE15': {'id': 'SE15', 'age': 64, 'height': 150, 'weight': 50, 'gender': 'F'}}
    #     subject = self.subject_data[subject_id]
    #     return Subject(subject['id'], subject['age'], subject['height'], subject['weight'], subject['gender'])
    #
    # def __activity_code_mapper(self, activity_code):
    #     self.activity_ids = {'D01': {'code': 'D01', 'fall': False, 'description': 'Walking slowly', 'trials': 1, 'duration': 100},
    #                      'D02': {'code': 'D02', 'fall': False, 'description': 'Walking quickly', 'trials': 1, 'duration': 100},
    #                      'D03': {'code': 'D03', 'fall': False, 'description': 'Jogging slowly', 'trials': 1, 'duration': 100},
    #                      'D04': {'code': 'D04', 'fall': False, 'description': 'Jogging quickly', 'trials': 1, 'duration': 100},
    #                      'D05': {'code': 'D05', 'fall': False, 'description': 'Walking upstairs and downstairs slowly', 'trials': 5, 'duration': 25},
    #                      'D06': {'code': 'D06', 'fall': False, 'description': 'Walking upstairs and downstairs quickly', 'trials': 5, 'duration': 25},
    #                      'D07': {'code': 'D07', 'fall': False, 'description': 'Slowly sit in a half height chair, wait a moment, and up slowly', 'trials': 5, 'duration': 12},
    #                      'D08': {'code': 'D08', 'fall': False, 'description': 'Quickly sit in a half height chair, wait a moment, and up quickly', 'trials': 5, 'duration': 12},
    #                      'D09': {'code': 'D09', 'fall': False, 'description': 'Slowly sit in a low height chair, wait a moment, and up slowly', 'trials': 5, 'duration': 12},
    #                      'D10': {'code': 'D10', 'fall': False, 'description': 'Quickly sit in a low height chair, wait a moment, and up quickly', 'trials': 5, 'duration': 12},
    #                      'D11': {'code': 'D11', 'fall': False, 'description': 'Sitting a moment, trying to get up, and collapse into a chair', 'trials': 5, 'duration': 12},
    #                      'D12': {'code': 'D12', 'fall': False, 'description': 'Sitting a moment, lying slowly, wait a moment, and sit again', 'trials': 5, 'duration': 12},
    #                      'D13': {'code': 'D13', 'fall': False, 'description': 'Sitting a moment, lying quickly, wait a moment, and sit again', 'trials': 5, 'duration': 12},
    #                      'D14': {'code': 'D14', 'fall': False, 'description': 'Being on one’s back change to lateral position, wait a moment, and change to one’s back', 'trials': 5, 'duration': 12},
    #                      'D15': {'code': 'D15', 'fall': False, 'description': 'Standing, slowly bending at knees, and getting up', 'trials': 5, 'duration': 12},
    #                      'D16': {'code': 'D16', 'fall': False, 'description': 'Standing, slowly bending without bending knees, and getting up', 'trials': 5, 'duration': 12},
    #                      'D17': {'code': 'D17', 'fall': False, 'description': 'Standing, get into a car, remain seated and get out of the car', 'trials': 5, 'duration': 12},
    #                      'D18': {'code': 'D18', 'fall': False, 'description': 'Stumble while walking', 'trials': 5, 'duration': 12},
    #                      'D19': {'code': 'D19', 'fall': False, 'description': 'Gently jump without falling (trying to reach a high object)', 'trials': 5, 'duration': 12},
    #                      'F01': {'code': 'F01', 'fall': True, 'description': 'Fall forward while walking caused by a slip', 'trials': 5, 'duration': 15},
    #                      'F02': {'code': 'F02', 'fall': True, 'description': 'Fall backward while walking caused by a slip', 'trials': 5, 'duration': 15},
    #                      'F03': {'code': 'F03', 'fall': True, 'description': 'Lateral fall while walking caused by a slip', 'trials': 5, 'duration': 15},
    #                      'F04': {'code': 'F04', 'fall': True, 'description': 'Fall forward while walking caused by a trip', 'trials': 5, 'duration': 15},
    #                      'F05': {'code': 'F05', 'fall': True, 'description': 'Fall forward while jogging caused by a trip', 'trials': 5, 'duration': 15},
    #                      'F06': {'code': 'F06', 'fall': True, 'description': 'Vertical fall while walking caused by fainting', 'trials': 5, 'duration': 15},
    #                      'F07': {'code': 'F07', 'fall': True, 'description': 'Fall while walking, with use of hands in a table to dampen fall, caused by fainting', 'trials': 5, 'duration': 15},
    #                      'F08': {'code': 'F08', 'fall': True, 'description': 'Fall forward when trying to get up', 'trials': 5, 'duration': 15},
    #                      'F09': {'code': 'F09', 'fall': True, 'description': 'Lateral fall when trying to get up', 'trials': 5, 'duration': 15},
    #                      'F10': {'code': 'F10', 'fall': True, 'description': 'Fall forward when trying to sit down', 'trials': 5, 'duration': 15},
    #                      'F11': {'code': 'F11', 'fall': True, 'description': 'Fall backward when trying to sit down', 'trials': 5, 'duration': 15},
    #                      'F12': {'code': 'F12', 'fall': True, 'description': 'Lateral fall when trying to sit down', 'trials': 5, 'duration': 15},
    #                      'F13': {'code': 'F13', 'fall': True, 'description': 'Fall forward while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15},
    #                      'F14': {'code': 'F14', 'fall': True, 'description': 'Fall backward while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15},
    #                      'F15': {'code': 'F15', 'fall': True, 'description': 'Lateral fall while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15}}
    #     activity = self.activity_ids[activity_code]
    #     return Activity(activity['code'], activity['fall'], activity['description'], activity['trials'], activity['duration'])


# class ConvertSisFallDataset:
#
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
#                         data = pd.read_csv(mfp, sep=',', header=None)
#                         data.columns = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
#                         data['MMA8451Q_z'] = [value.replace(';', '') for value in data['MMA8451Q_z']]
#                         data = self.__apply_unit_conversion_to_all(data)
#                         # Add time column to the data, all sensors recoding at 200 Hz,
#                         # all sensors record same number of samples
#                         data['time'] = self.__make_time_array(200, len(data['ADXL345_x']))
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
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\SisFall_csv\SisFall_dataset_csv'
    db = DatasetBuilder()
    db.generate_data_file_paths(path)

if __name__ == '__main__':
    main()
