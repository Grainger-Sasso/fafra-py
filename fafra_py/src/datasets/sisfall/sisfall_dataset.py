import os
import pandas as pd
import numpy as np
from typing import List
from fafra_py_legacy.src.dataset_tools.params.motion_data import MotionData
from fafra_py_legacy.src.dataset_tools.params.motion_dataset import MotionDataset
from fafra_py_legacy.src.dataset_tools.params.sensor import Sensor
from fafra_py_legacy.src.dataset_tools.params.subject import Subject
from fafra_py_legacy.src.dataset_tools.params.activity import Activity
from fafra_py_legacy.src.dataset_tools.motion_data.acceleration.linear_acceleration.linear_acceleration import LinearAcceleration
from fafra_py_legacy.src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
from fafra_py_legacy.src.dataset_tools.motion_data.acceleration.angular_acceleration.angular_acceleration import AngularAcceleration
from fafra_py_legacy.src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration


class SisFallDataset(MotionDataset):

    def __init__(self, path, file_format):
        super().__init__('SisFall', path, file_format, 'activity', 'subject', 200.0,
                         {'ADXL345': Sensor('ADXL345', 'accelerometer', 200, 13, 16),
                          'ITG3200': Sensor('ITG3200', 'gyroscope', 200, 16, 2000),
                          'MMA8451Q': Sensor('MMA8451Q', 'accelerometer', 200, 14, 8)})

    def read_dataset(self):
        # Initialize output
        all_motion_data_in_dataset: List[MotionData] = []
        # Iterate through SisFall directory for all data files
        for root, dirs, files in list(os.walk(self.path))[1:]:
            for file in files:
                if file == 'Readme.txt' or not file.endswith('.' + self.file_format):
                    continue
                else:
                    motion_file_path = os.path.join(root, file)
                    print(motion_file_path)
                    # Open file, read in txt file as csv data
                    with open(motion_file_path) as mfp:
                        data = pd.read_csv(mfp, sep=',', index_col='time')
                        all_motion_data_in_dataset.append(
                            self.__build_motion_data(data, file))
        self.motion_data = all_motion_data_in_dataset

    def write_dataset_to_csv(self, output_folder_path):
        output_path = os.path.join(output_folder_path, 'SisFall_dataset_csv')
        for motion_data in self.motion_data:
            print(motion_data.subject.id)
            subject_path = os.path.join(output_path, motion_data.subject.id)
            os.makedirs(subject_path, exist_ok=True)
            filename = motion_data.activity.code + '_' + motion_data.subject.id + '_' + motion_data.trial + '.csv'
            file_path = os.path.join(subject_path, filename)
            motion_data.motion_df.to_csv(file_path, header=True, index=False)

    def __build_motion_data(self, data, file):
        # Build out acceleration instances from data
        adxl_x = LinearAcceleration('X', 'Frontal', np.array(data['ADXL345_x']), np.array(data.index), self.sensor_data['ADXL345'])
        adxl_y = LinearAcceleration('Y', 'Vertical', np.array(data['ADXL345_y']), np.array(data.index), self.sensor_data['ADXL345'])
        adxl_z = LinearAcceleration('Z', 'Sagittal', np.array(data['ADXL345_z']), np.array(data.index), self.sensor_data['ADXL345'])
        adxl_tri_acc = TriaxialLinearAcceleration('Triaxial Linear', adxl_x, adxl_y, adxl_z)
        # Build out acceleration instances from data
        itg_x = AngularAcceleration('X', 'Frontal', np.array(data['ITG3200_x']), np.array(data.index), self.sensor_data['ITG3200'])
        itg_y = AngularAcceleration('Y', 'Vertical', np.array(data['ITG3200_y']), np.array(data.index), self.sensor_data['ITG3200'])
        itg_z = AngularAcceleration('Z', 'Sagittal', np.array(data['ITG3200_z']), np.array(data.index), self.sensor_data['ITG3200'])
        itg_tri_acc = TriaxialAngularAcceleration('Triaxial Angular', itg_x, itg_y, itg_z)
        # Build out acceleration instances from data
        mma_x = LinearAcceleration('X', 'Frontal', np.array(data['MMA8451Q_x']), np.array(data.index), self.sensor_data['MMA8451Q'])
        mma_y = LinearAcceleration('Y', 'Vertical', np.array(data['MMA8451Q_y']), np.array(data.index), self.sensor_data['MMA8451Q'])
        mma_z = LinearAcceleration('Z', 'Sagittal', np.array(data['MMA8451Q_z']), np.array(data.index), self.sensor_data['MMA8451Q'])
        mma_tri_acc = TriaxialLinearAcceleration('Triaxial Linear', mma_x, mma_y, mma_z)
        # Get activity code, subject id, and trial from file name scheme
        activity_code, subject_id, trial = os.path.splitext(file)[0].split('_')
        subject = self.__subject_id_mapper(subject_id)
        activity = self.__activity_code_mapper(activity_code)
        # Build motion data
        motion_data = MotionData(subject, activity, trial, data,
                                 [adxl_tri_acc, mma_tri_acc], [itg_tri_acc])
        return motion_data

    def __subject_id_mapper(self, subject_id):
        self.subject_data = {'SA01': {'id': 'SA01', 'age': 26, 'height': 165, 'weight': 53, 'gender': 'F'},
                  'SA02': {'id': 'SA02', 'age': 23, 'height': 176, 'weight': 58.5, 'gender': 'M'},
                  'SA03': {'id': 'SA03', 'age': 19, 'height': 156, 'weight': 48, 'gender': 'F'},
                  'SA04': {'id': 'SA04', 'age': 23, 'height': 170, 'weight': 72, 'gender': 'M'},
                  'SA05': {'id': 'SA05', 'age': 22, 'height': 172, 'weight': 69.5, 'gender': 'M'},
                  'SA06': {'id': 'SA06', 'age': 21, 'height': 169, 'weight': 58, 'gender': 'M'},
                  'SA07': {'id': 'SA07', 'age': 21, 'height': 156, 'weight': 63, 'gender': 'F'},
                  'SA08': {'id': 'SA08', 'age': 21, 'height': 149, 'weight': 41.5, 'gender': 'F'},
                  'SA09': {'id': 'SA09', 'age': 24, 'height': 165, 'weight': 64, 'gender': 'M'},
                  'SA10': {'id': 'SA10', 'age': 21, 'height': 177, 'weight': 67, 'gender': 'M'},
                  'SA11': {'id': 'SA11', 'age': 19, 'height': 170, 'weight': 80.5, 'gender': 'M'},
                  'SA12': {'id': 'SA12', 'age': 25, 'height': 153, 'weight': 47, 'gender': 'F'},
                  'SA13': {'id': 'SA13', 'age': 22, 'height': 157, 'weight': 55, 'gender': 'F'},
                  'SA14': {'id': 'SA14', 'age': 27, 'height': 160, 'weight': 46, 'gender': 'F'},
                  'SA15': {'id': 'SA15', 'age': 25, 'height': 160, 'weight': 52, 'gender': 'F'},
                  'SA16': {'id': 'SA16', 'age': 20, 'height': 169, 'weight': 61, 'gender': 'F'},
                  'SA17': {'id': 'SA17', 'age': 23, 'height': 182, 'weight': 75, 'gender': 'M'},
                  'SA18': {'id': 'SA18', 'age': 23, 'height': 181, 'weight': 73, 'gender': 'M'},
                  'SA19': {'id': 'SA19', 'age': 30, 'height': 170, 'weight': 76, 'gender': 'M'},
                  'SA20': {'id': 'SA20', 'age': 30, 'height': 150, 'weight': 42, 'gender': 'F'},
                  'SA21': {'id': 'SA21', 'age': 30, 'height': 183, 'weight': 68, 'gender': 'M'},
                  'SA22': {'id': 'SA22', 'age': 19, 'height': 158, 'weight': 50.5, 'gender': 'F'},
                  'SA23': {'id': 'SA23', 'age': 24, 'height': 156, 'weight': 48, 'gender': 'F'},
                  'SE01': {'id': 'SE01', 'age': 71, 'height': 171, 'weight': 102, 'gender': 'M'},
                  'SE02': {'id': 'SE02', 'age': 75, 'height': 150, 'weight': 57, 'gender': 'F'},
                  'SE03': {'id': 'SE03', 'age': 62, 'height': 150, 'weight': 51, 'gender': 'F'},
                  'SE04': {'id': 'SE04', 'age': 63, 'height': 160, 'weight': 59, 'gender': 'F'},
                  'SE05': {'id': 'SE05', 'age': 63, 'height': 165, 'weight': 72, 'gender': 'M'},
                  'SE06': {'id': 'SE06', 'age': 60, 'height': 163, 'weight': 79, 'gender': 'M'},
                  'SE07': {'id': 'SE07', 'age': 65, 'height': 168, 'weight': 76, 'gender': 'M'},
                  'SE08': {'id': 'SE08', 'age': 68, 'height': 163, 'weight': 72, 'gender': 'F'},
                  'SE09': {'id': 'SE09', 'age': 66, 'height': 167, 'weight': 65, 'gender': 'M'},
                  'SE10': {'id': 'SE10', 'age': 64, 'height': 156, 'weight': 66, 'gender': 'F'},
                  'SE11': {'id': 'SE11', 'age': 66, 'height': 169, 'weight': 63, 'gender': 'F'},
                  'SE12': {'id': 'SE12', 'age': 69, 'height': 164, 'weight': 56.5, 'gender': 'M'},
                  'SE13': {'id': 'SE13', 'age': 65, 'height': 171, 'weight': 72.5, 'gender': 'M'},
                  'SE14': {'id': 'SE14', 'age': 67, 'height': 163, 'weight': 58, 'gender': 'M'},
                  'SE15': {'id': 'SE15', 'age': 64, 'height': 150, 'weight': 50, 'gender': 'F'}}
        subject = self.subject_data[subject_id]
        return Subject(subject['id'], subject['age'], subject['height'], subject['weight'], subject['gender'])

    def __activity_code_mapper(self, activity_code):
        self.activity_ids = {'D01': {'code': 'D01', 'fall': False, 'description': 'Walking slowly', 'trials': 1, 'duration': 100},
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
                         'F15': {'code': 'F15', 'fall': True, 'description': 'Lateral fall while sitting, caused by fainting or falling asleep', 'trials': 5, 'duration': 15}}
        activity = self.activity_ids[activity_code]
        return Activity(activity['code'], activity['fall'], activity['description'], activity['trials'], activity['duration'])


class ConvertSisFallDataset:

    def __init__(self):
        self.sisfall = SisFallDataset(r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_py\Fall Datasets\SisFall\SisFall_dataset',
                                      'txt')

    # TODO: add functionality to generate csv files from txt files in original dataset
    def convert_text_files_to_csv(self):
        # Read in txt files

        # Convert data and build objects
        # Write to csv
        pass

    def read_dataset(self):
        # Initialize output
        all_motion_data_in_dataset: List[MotionData] = []
        # Iterate through SisFall directory for all data files
        for root, dirs, files in list(os.walk(self.path))[1:]:
            for file in files:
                if file == 'Readme.txt' or not file.endswith('.' + self.file_format):
                    continue
                else:
                    motion_file_path = os.path.join(root, file)
                    print(motion_file_path)
                    # Open file, read in txt file as csv data
                    with open(motion_file_path) as mfp:
                        data = pd.read_csv(mfp, sep=',', header=None)
                        data.columns = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
                        data['MMA8451Q_z'] = [value.replace(';', '') for value in data['MMA8451Q_z']]
                        data = self.__apply_unit_conversion_to_all(data)
                        # Add time column to the data, all sensors recoding at 200 Hz,
                        # all sensors record same number of samples
                        data['time'] = self.__make_time_array(200, len(data['ADXL345_x']))
                        all_motion_data_in_dataset.append(
                            self.__build_motion_data(data, file))
        self.motion_data = all_motion_data_in_dataset

    def write_dataset_to_csv(self, output_folder_path):
        output_path = os.path.join(output_folder_path, 'SisFall_dataset_csv')
        for motion_data in self.motion_data:
            subject_path = os.path.join(output_path, motion_data.subject.id)
            os.makedirs(subject_path, exist_ok=True)
            filename = motion_data.activity.code + '_' + motion_data.subject.id + '_' + motion_data.trial + '.csv'
            file_path = os.path.join(subject_path, filename)
            motion_data.motion_df.to_csv(file_path, header=True)

    def __apply_unit_conversion_to_all(self, data):
        data["ADXL345_x"] = self.__apply_unit_conversion(np.array(data['ADXL345_x'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ADXL345'])
        data["ADXL345_y"] = self.__apply_unit_conversion(np.array(data['ADXL345_y'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ADXL345'])
        data["ADXL345_z"] = self.__apply_unit_conversion(np.array(data['ADXL345_z'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ADXL345'])
        data["ITG3200_x"] = self.__apply_unit_conversion(np.array(data['ITG3200_x'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ITG3200'])
        data["ITG3200_y"] = self.__apply_unit_conversion(np.array(data['ITG3200_y'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ITG3200'])
        data["ITG3200_z"] = self.__apply_unit_conversion(np.array(data['ITG3200_z'],
                                                                  dtype=np.float),
                                                         self.sensor_data[
                                                             'ITG3200'])
        data["MMA8451Q_x"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_x'],
                                                                   dtype=np.float),
                                                          self.sensor_data[
                                                              'MMA8451Q'])
        data["MMA8451Q_y"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_y'],
                                                                   dtype=np.float),
                                                          self.sensor_data[
                                                              'MMA8451Q'])
        data["MMA8451Q_z"] = self.__apply_unit_conversion(np.array(data['MMA8451Q_z'],
                                                                   dtype=np.float),
                                                          self.sensor_data[
                                                              'MMA8451Q'])
        return data

    def __apply_unit_conversion(self, single_acc_axis, sensor: Sensor) -> np.array:
        # Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
        # Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
        return np.array(single_acc_axis * ((2*sensor.range)/(2**sensor.resolution)))

    def __make_time_array(self, sample_rate: int, number_of_samples: int):
        recording_time = (number_of_samples-1)/sample_rate
        time_array = np.linspace(0.0, recording_time, number_of_samples)
        return time_array



def main():
    sisfall_dataset = SisFallDataset(r'', '')

if __name__ == '__main__':
    main()
