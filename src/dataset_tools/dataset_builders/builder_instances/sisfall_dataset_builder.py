import os
import json
import pandas as pd
import numpy as np
import glob
import time

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
        # TODO: add second sisfall dataset for the second accelerometer in dataset, currently not being used
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 200.0
        # Original units: g,g,g,°/s,°/s,°/s
        # Converted to: m/s^2,m/s^2,m/s^2,°/s,°/s,°/s
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        # Adults were not screened for fall risk, therefore none of them are assumed to be fallers
        self.subject_data = {
            'SA01': {'id': 'SA01', 'age': 26, 'height': 165, 'weight': 53, 'sex': 'F'},
            'SA02': {'id': 'SA02', 'age': 23, 'height': 176, 'weight': 58.5, 'sex': 'M'},
            'SA03': {'id': 'SA03', 'age': 19, 'height': 156, 'weight': 48, 'sex': 'F'},
            'SA04': {'id': 'SA04', 'age': 23, 'height': 170, 'weight': 72, 'sex': 'M'},
            'SA05': {'id': 'SA05', 'age': 22, 'height': 172, 'weight': 69.5, 'sex': 'M'},
            'SA06': {'id': 'SA06', 'age': 21, 'height': 169, 'weight': 58, 'sex': 'M'},
            'SA07': {'id': 'SA07', 'age': 21, 'height': 156, 'weight': 63, 'sex': 'F'},
            'SA08': {'id': 'SA08', 'age': 21, 'height': 149, 'weight': 41.5, 'sex': 'F'},
            'SA09': {'id': 'SA09', 'age': 24, 'height': 165, 'weight': 64, 'sex': 'M'},
            'SA10': {'id': 'SA10', 'age': 21, 'height': 177, 'weight': 67, 'sex': 'M'},
            'SA11': {'id': 'SA11', 'age': 19, 'height': 170, 'weight': 80.5, 'sex': 'M'},
            'SA12': {'id': 'SA12', 'age': 25, 'height': 153, 'weight': 47, 'sex': 'F'},
            'SA13': {'id': 'SA13', 'age': 22, 'height': 157, 'weight': 55, 'sex': 'F'},
            'SA14': {'id': 'SA14', 'age': 27, 'height': 160, 'weight': 46, 'sex': 'F'},
            'SA15': {'id': 'SA15', 'age': 25, 'height': 160, 'weight': 52, 'sex': 'F'},
            'SA16': {'id': 'SA16', 'age': 20, 'height': 169, 'weight': 61, 'sex': 'F'},
            'SA17': {'id': 'SA17', 'age': 23, 'height': 182, 'weight': 75, 'sex': 'M'},
            'SA18': {'id': 'SA18', 'age': 23, 'height': 181, 'weight': 73, 'sex': 'M'},
            'SA19': {'id': 'SA19', 'age': 30, 'height': 170, 'weight': 76, 'sex': 'M'},
            'SA20': {'id': 'SA20', 'age': 30, 'height': 150, 'weight': 42, 'sex': 'F'},
            'SA21': {'id': 'SA21', 'age': 30, 'height': 183, 'weight': 68, 'sex': 'M'},
            'SA22': {'id': 'SA22', 'age': 19, 'height': 158, 'weight': 50.5, 'sex': 'F'},
            'SA23': {'id': 'SA23', 'age': 24, 'height': 156, 'weight': 48, 'sex': 'F'},
            'SE01': {'id': 'SE01', 'age': 71, 'height': 171, 'weight': 102, 'sex': 'M'},
            'SE02': {'id': 'SE02', 'age': 75, 'height': 150, 'weight': 57, 'sex': 'F'},
            'SE03': {'id': 'SE03', 'age': 62, 'height': 150, 'weight': 51, 'sex': 'F'},
            'SE04': {'id': 'SE04', 'age': 63, 'height': 160, 'weight': 59, 'sex': 'F'},
            'SE05': {'id': 'SE05', 'age': 63, 'height': 165, 'weight': 72, 'sex': 'M'},
            'SE06': {'id': 'SE06', 'age': 60, 'height': 163, 'weight': 79, 'sex': 'M'},
            'SE07': {'id': 'SE07', 'age': 65, 'height': 168, 'weight': 76, 'sex': 'M'},
            'SE08': {'id': 'SE08', 'age': 68, 'height': 163, 'weight': 72, 'sex': 'F'},
            'SE09': {'id': 'SE09', 'age': 66, 'height': 167, 'weight': 65, 'sex': 'M'},
            'SE10': {'id': 'SE10', 'age': 64, 'height': 156, 'weight': 66, 'sex': 'F'},
            'SE11': {'id': 'SE11', 'age': 66, 'height': 169, 'weight': 63, 'sex': 'F'},
            'SE12': {'id': 'SE12', 'age': 69, 'height': 164, 'weight': 56.5, 'sex': 'M'},
            'SE13': {'id': 'SE13', 'age': 65, 'height': 171, 'weight': 72.5, 'sex': 'M'},
            'SE14': {'id': 'SE14', 'age': 67, 'height': 163, 'weight': 58, 'sex': 'M'},
            'SE15': {'id': 'SE15', 'age': 64, 'height': 150, 'weight': 50, 'sex': 'F'}
        }
        self.activity_codes = {
            'D01': {'code': 'D01', 'fall': False, 'description': 'Walking slowly', 'trials': 1, 'duration': 100},
            'D02': {'code': 'D02', 'fall': False, 'description': 'Walking quickly', 'trials': 1, 'duration': 100},
            'D03': {'code': 'D03', 'fall': False, 'description': 'Jogging slowly', 'trials': 1, 'duration': 100},
            'D04': {'code': 'D04', 'fall': False, 'description': 'Jogging quickly', 'trials': 1, 'duration': 100},
            'D05': {'code': 'D05', 'fall': False, 'description': 'Walking upstairs and downstairs slowly', 'trials': 5, 'duration': 25},
            'D06': {'code': 'D06', 'fall': False, 'description': 'Walking upstairs and downstairs quickly', 'trials': 5, 'duration': 25},
            'D07': {'code': 'D07', 'fall': False, 'description': 'Slowly sit in a half height chair, wait a moment, and up slowly', 'trials': 5, 'duration': 12},
            'D08': {'code': 'D08', 'fall': False, 'description': 'Quickly sit in a half height chair, wait a moment, and up quickly', 'trials': 5, 'duration': 12},
            'D09': {'code': 'D09', 'fall': False, 'description': 'Slowly sit in a low height chair, wait a moment, and up slowly', 'trials': 5, 'duration': 12},
            'D10': {'code': 'D10', 'fall': False, 'description': 'Quickly sit in a low height chair, wait a moment, and up quickly', 'trials': 5, 'duration': 12},
            'D11': {'code': 'D11', 'fall': False, 'description': 'Sitting a moment, trying to get up, and collapse into a chair', 'trials': 5, 'duration': 12},
            'D12': {'code': 'D12', 'fall': False, 'description': 'Sitting a moment, lying slowly, wait a moment, and sit again', 'trials': 5, 'duration': 12},
            'D13': {'code': 'D13', 'fall': False, 'description': 'Sitting a moment, lying quickly, wait a moment, and sit again', 'trials': 5, 'duration': 12},
            'D14': {'code': 'D14', 'fall': False, 'description': 'Being on one’s back change to lateral position, wait a moment, and change to one’s back', 'trials': 5, 'duration': 12},
            'D15': {'code': 'D15', 'fall': False, 'description': 'Standing, slowly bending at knees, and getting up', 'trials': 5, 'duration': 12},
            'D16': {'code': 'D16', 'fall': False, 'description': 'Standing, slowly bending without bending knees, and getting up', 'trials': 5, 'duration': 12},
            'D17': {'code': 'D17', 'fall': False, 'description': 'Standing, get into a car, remain seated and get out of the car', 'trials': 5, 'duration': 12},
            'D18': {'code': 'D18', 'fall': False, 'description': 'Stumble while walking', 'trials': 5, 'duration': 12},
            'D19': {'code': 'D19', 'fall': False, 'description': 'Gently jump without falling (trying to reach a high object)', 'trials': 5, 'duration': 12},
            'F01': {'code': 'F01', 'fall': True, 'description': 'Fall forward while walking caused by a slip', 'trials': 5, 'duration': 15},
            'F02': {'code': 'F02', 'fall': True, 'description': 'Fall backward while walking caused by a slip', 'trials': 5, 'duration': 15},
            'F03': {'code': 'F03', 'fall': True, 'description': 'Lateral fall while walking caused by a slip', 'trials': 5, 'duration': 15},
            'F04': {'code': 'F04', 'fall': True, 'description': 'Fall forward while walking caused by a trip', 'trials': 5, 'duration': 15},
            'F05': {'code': 'F05', 'fall': True, 'description': 'Fall forward while jogging caused by a trip', 'trials': 5, 'duration': 15},
            'F06': {'code': 'F06', 'fall': True, 'description': 'Vertical fall while walking caused by fainting', 'trials': 5, 'duration': 15},
            'F07': {'code': 'F07', 'fall': True, 'description': 'Fall while walking, with use of hands in a table to dampen fall, caused by fainting', 'trials': 5, 'duration': 15},
            'F08': {'code': 'F08', 'fall': True, 'description': 'Fall forward when trying to get up', 'trials': 5, 'duration': 15},
            'F09': {'code': 'F09', 'fall': True, 'description': 'Lateral fall when trying to get up', 'trials': 5, 'duration': 15},
            'F10': {'code': 'F10', 'fall': True, 'description': 'Fall forward when trying to sit down', 'trials': 5, 'duration': 15},
            'F11': {'code': 'F11', 'fall': True, 'description': 'Fall backward when trying to sit down', 'trials': 5, 'duration': 15},
            'F12': {'code': 'F12', 'fall': True, 'description': 'Lateral fall when trying to sit down', 'trials': 5, 'duration': 15},
            'F13': {'code': 'F13', 'fall': True, 'description': 'Fall forward while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15},
            'F14': {'code': 'F14', 'fall': True, 'description': 'Fall backward while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15},
            'F15': {'code': 'F15', 'fall': True, 'description': 'Lateral fall while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15}
        }

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        data_file_paths = self._generate_data_file_paths(dataset_path)
        dataset = []
        for subj_id, data_file_paths in data_file_paths.items():
            print(subj_id)
            for file_path in data_file_paths:
                data = self._read_data_file(file_path)
                # Convert accelerometer data from g to m/s^2
                data[:, 0:3] = data[:, 0:3] * 9.80665
                imu_data_file_path: str = file_path
                imu_data_file_name: str = \
                os.path.split(os.path.splitext(imu_data_file_path)[0])[1]
                imu_metadata_file_path: str = 'N/A'
                imu_metadata = IMUMetadata(None,
                                           self.sampling_frequency, self.units)
                subj_clin_data = self._get_subj_clin_data(subj_id)
                # Build dataset objects from the data and metadata
                if segment_dataset:
                    # TODO: track the segmented data with a linked list
                    # Segment the data and build a UserData object for each epoch
                    data_segments = self.segment_data(data, epoch_size,
                                                      self.sampling_frequency)
                    for segment in data_segments:
                        imu_data = self._generate_imu_data_instance(file_path, segment,
                                                                    self.sampling_frequency)
                        dataset.append(UserData(imu_data_file_path,
                                                imu_data_file_name,
                                                imu_metadata_file_path,
                                                clinical_demo_path,
                                                {
                                                    IMUDataFilterType.RAW: imu_data},
                                                imu_metadata,
                                                subj_clin_data))
                else:
                    # Build a UserData object for the whole data
                    imu_data = self._generate_imu_data_instance(file_path, data,
                                                                self.sampling_frequency)
                    dataset.append(
                        UserData(imu_data_file_path,
                                 imu_data_file_name,
                                 imu_metadata_file_path,
                                 clinical_demo_path,
                                 {IMUDataFilterType.RAW: imu_data},
                                 imu_metadata, subj_clin_data))
        return Dataset(self.get_dataset_name(), dataset_path, clinical_demo_path, dataset, self.activity_codes)

    def write_csv_dataset_to_json(self, dataset_path, samp_freq, output_dir):
        # Read in the data from CSV files
        data_file_paths = self._generate_data_file_paths(dataset_path)
        dataset = []
        for subj_id, data_file_paths in data_file_paths.items():
            out_subj_dir = os.path.join(output_dir, subj_id)
            if not os.path.exists(out_subj_dir):
                # Create output dir for the subject
                os.mkdir(out_subj_dir)
            for file_path in data_file_paths:
                data = self._read_data_file(file_path)
                json_data = {}
                json_data['v_acc_data'] = np.array(data[1])
                json_data['ml_acc_data'] = np.array(data[0])
                json_data['ap_acc_data'] = np.array(data[2])
                json_data['yaw_gyr_data'] = np.array(data[4])
                json_data['pitch_gyr_data'] = np.array(data[3])
                json_data['roll_gyr_data'] = np.array(data[5])
                json_data['time'] = np.linspace(0, len(np.array(data[1])) / int(samp_freq), len(np.array(data[1])))
                # Make file for
                # Create output path
                json_path = out_subj_dir + os.path.splitext(os.path.basename(file_path))[0] + '.json'
                # Write the data out to JSON file
                with open(json_path, 'w') as jf:
                    json.dump(json_data, jf)

    def _generate_data_file_paths(self, dataset_path):
        data_file_paths = {}
        # Iterate through all of the files in the CSV directory, get all filenames
        for subject_id in next(os.walk(dataset_path))[1]:
            data_file_paths[subject_id] = []
            subj_data_folder = os.path.join(dataset_path, subject_id)
            for data_file_path in glob.glob(
                os.path.join(subj_data_folder, '*.csv')):
                data_file_paths[subject_id].append(data_file_path)
        return data_file_paths

    def _read_data_file(self, data_file_path):
        # Reading csv file into numpy array,
        # ignoring second accelerometer for now
        with open(data_file_path) as mfp:
            data = pd.read_csv(mfp, sep=',', index_col='time')
            data = data.to_numpy().T
            return data[1:7].T

    def _get_subj_clin_data(self, subj_id):
        subj_data = self.subject_data[subj_id]
        trial = ''
        return ClinicalDemographicData(subj_data['id'], subj_data['age'], subj_data['sex'], False, float(subj_data['height']), trial)

    def _generate_imu_data_instance(self, file_path, data, samp_freq):
        # Positive x: right, mediolateral
        # Positive y: down, vertical
        # Positive z: forward, anteroposterior
        # Data: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
        activity_code = os.path.split(file_path)[1][0:3]
        activity_description = self.activity_codes[activity_code]['description']
        v_acc_data = np.array(data.T[1])
        # Flip the direction of vertical axis data such that gravity is now positive
        v_acc_data = v_acc_data * -1.0
        ml_acc_data = np.array(data.T[0])
        ap_acc_data = np.array(data.T[2])
        yaw_gyr_data = np.array(data.T[4])
        pitch_gyr_data = np.array(data.T[3])
        roll_gyr_data = np.array(data.T[5])
        time = np.linspace(0, len(v_acc_data) / int(samp_freq),
                           len(v_acc_data))
        return IMUData(activity_code, activity_description, v_acc_data,
                       ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                       roll_gyr_data, time)


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
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\SisFall_csv\SisFall_dataset_csv'
    t0 = time.time()
    db = DatasetBuilder()
    dataset = db.build_dataset(path, 'N/A', True, 8.0)
    print(str(time.time() - t0))
    print(dataset)


if __name__ == '__main__':
    main()
