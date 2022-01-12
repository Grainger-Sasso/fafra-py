import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Any
from matplotlib import pyplot as plt

from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


DATASET_NAME = DatasetNames.UIUC_GAIT


class DatasetBuilder(DatasetBuilder):
    def __init__(self):
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 64.0
        # Original units: g,g,g
        # Converted to: m/s^2,m/s^2,m/s^2
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': 'N/A', 'pitch': 'N/A', 'roll': 'N/A'}
        self.subject_data = {
            '101': {'id': '101', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '102': {'id': '102', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '104': {'id': '104', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '105': {'id': '105', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '106': {'id': '106', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '107': {'id': '107', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '108': {'id': '108', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '109': {'id': '109', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '110': {'id': '110', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '111': {'id': '111', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '112': {'id': '112', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '113': {'id': '113', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '114': {'id': '114', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '115': {'id': '115', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '201': {'id': '201', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '202': {'id': '202', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '203': {'id': '203', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '204': {'id': '204', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '205': {'id': '205', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '206': {'id': '206', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '207': {'id': '207', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '208': {'id': '208', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '209': {'id': '209', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '210': {'id': '210', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '211': {'id': '211', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '212': {'id': '212', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '213': {'id': '213', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '214': {'id': '214', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '215': {'id': '215', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '216': {'id': '216', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '217': {'id': '217', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '219': {'id': '219', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '220': {'id': '220', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '221': {'id': '221', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '222': {'id': '222', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '223': {'id': '223', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '224': {'id': '224', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '225': {'id': '225', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '227': {'id': '227', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '228': {'id': '228', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '229': {'id': '229', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '230': {'id': '230', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '231': {'id': '231', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '301': {'id': '301', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '302': {'id': '302', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '304': {'id': '304', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '305': {'id': '305', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '306': {'id': '306', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '307': {'id': '307', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '309': {'id': '309', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '310': {'id': '308', 'age': None, 'height': None, 'weight': None,
                    'sex': None},
            '311': {'id': '311', 'age': None, 'height': None, 'weight': None,
                    'sex': None}

        }
        self.activity_codes = {}

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        data_file_paths = self._generate_data_file_paths(dataset_path)
        dataset: List[UserData] = []
        for subj in data_file_paths:
            subj_id = subj['subj_id']
            print(subj_id)
            for trial in subj['trials']:
                trial_id = trial['trial_id']
                paths = trial['acc_paths']
                imu_metadata_file_path = trial['metadata_path']
                # Read in acc data files and metadata files
                # only if number of acc paths for trial is 3
                if len(paths) == 3:
                    # Read in imu metadata
                    raw_imu_metadata = self._read_imu_metadata(imu_metadata_file_path)
                    # Read in the paths and build data objects
                    tri_ax_acc_data = self._read_data_files(paths)
                    imu_data_file_path: List[str] = paths
                    imu_data_file_name: str = 'acc_data'
                    imu_metadata = IMUMetadata(raw_imu_metadata,
                                               self.sampling_frequency,
                                               self.units)
                    subj_clin_data = self._get_subj_clin_data(subj_id)
                    data = np.array([
                        tri_ax_acc_data['x_acc_data'],
                        tri_ax_acc_data['y_acc_data'],
                        tri_ax_acc_data['z_acc_data']
                    ]).T
                    if segment_dataset:
                        # TODO: track the segmented data with a linked list
                        # Segment the data and build a UserData object for each epoch
                        data_segments = self.segment_data(data, epoch_size,
                                                          self.sampling_frequency)
                        for segment in data_segments:
                            imu_data = self._generate_imu_data_instance(
                                paths, segment,
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
                        imu_data = self._generate_imu_data_instance(
                            data, self.sampling_frequency)
                        dataset.append(
                            UserData(imu_data_file_path,
                                     imu_data_file_name,
                                     imu_metadata_file_path,
                                     clinical_demo_path,
                                     {IMUDataFilterType.RAW: imu_data},
                                     imu_metadata, subj_clin_data))
        return Dataset(self.get_dataset_name(), dataset_path,
                       clinical_demo_path, dataset,
                       self.activity_codes)

    def _generate_data_file_paths(self, dataset_path):
        """
        Returns the subject IDs, trial IDs, and associated accelerometer files
        in JSON format:
        [
          {
            "subj_id": XXX,
            "trials": [
              {
                "trial_id": XXXX,
                "paths": [
                  *ALL PATHS WITH ACCELERATION IN THEM*
                ]
              },...
            ]
          },...
        ]
        :param dataset_path:
        :return:
        """
        # Initialize the output variable, dir with subj and trial info + paths
        data_file_paths = []
        # Collects all subject IDs in the dataset based on folder inside
        # top dataset dir location
        subj_ids = next(os.walk(dataset_path))[1]
        # Assmebles paths to subj trial data from top dir level and subj IDs
        subj_paths = [os.path.join(dataset_path, subj_id) for subj_id in subj_ids]
        # Scrape all subj paths for trial IDs by folder names
        all_trial_ids = [next(os.walk(subj_path))[1] for subj_path in subj_paths]
        # Iterate through all trial folders searching for acceleration files
        for subj_path, subj_id, trial_ids in zip(subj_paths, subj_ids, all_trial_ids):
            subj_data_paths = {'subj_id': subj_id, 'trials': []}
            for trial_id in trial_ids:
                trial_data_paths = {'trial_id': trial_id, 'acc_paths': [], 'metadata_path': ''}
                # Search for acceleration files and construct their paths
                trial_path = os.path.join(subj_path, trial_id)
                metadata_path_found = False
                for root, dirs, files in os.walk(trial_path):
                    for file in files:
                        if 'acceleration' in file and file.endswith('.wav'):
                            trial_data_paths['acc_paths'].append(
                                os.path.join(root, file))
                        elif 'device_position.csv' in file:
                            trial_data_paths['metadata_path'] = os.path.join(root, file)
                            metadata_path_found = True
                num_files_found = len(trial_data_paths['acc_paths'])
                # If all the acceleration files were found and the metadata path was not found
                if num_files_found == 3 and not metadata_path_found:
                    raise ValueError(f'No metadata file path found for subj: {subj_id} and trial {trial_id}')
                subj_data_paths['trials'].append(trial_data_paths)
            data_file_paths.append(subj_data_paths)
        # Remove duplicated files from the first subject
        data_file_paths[0]['trials'][0]['acc_paths'] = data_file_paths[0]['trials'][0]['acc_paths'][3:]
        return data_file_paths

    def _read_imu_metadata(self, imu_metadata_file_path):
        md_df = pd.read_csv(imu_metadata_file_path)
        pos_key = md_df.keys()[1]
        dev_pos = md_df[pos_key][0]
        return dev_pos

    def _read_data_files(self, paths: List[str]):
        # Initialize output variable
        tri_ax_acc_data = {
            'x_acc_data': [],
            'y_acc_data': [],
            'z_acc_data': []
                           }
        for path in paths:
            data = self._read_wav(path)
            if path.lower()[-5] == 'x':
                tri_ax_acc_data['x_acc_data'] = data
            elif path.lower()[-5] == 'y':
                tri_ax_acc_data['y_acc_data'] = data
            elif path.lower()[-5] == 'z':
                tri_ax_acc_data['z_acc_data'] = data
            else:
                raise ValueError(f'Path does not contain axis info: {path}')
        return tri_ax_acc_data

    def _read_wav(self, path):
        # Data is stored as 16-bit int, converts to float
        data = wavfile.read(path)[1]
        data = np.array(data, dtype=float)
        #Initial units are g = value/256, converts directly from g to m/s^2
        data = data / 256.0 * 9.80665
        return data

    def _get_subj_clin_data(self, subj_id):
        subj_data = self.subject_data[subj_id]
        # TODO: replace this height with builder attribute for height
        # float(subj_data['height']
        subj_height = 2.0
        return ClinicalDemographicData(subj_data['id'], subj_data['age'], subj_data['sex'], False, subj_height)

    def _generate_imu_data_instance(self, data, samp_freq):
        """
        Positive x: forward, anteroposterior
        Positive y: down, vertical
        Positive z: right, mediolateral
        Data: acc_x (0 ix), acc_y (1 ix), acc_z (2 ix)
        """
        activity_code = 'walk'
        activity_description = 'walking at self-selected pace'
        v_acc_data = np.array(data.T[1])
        # Flip the direction of vertical axis data such that gravity is
        # now positive
        v_acc_data = v_acc_data * -1.0
        ml_acc_data = np.array(data.T[2])
        ap_acc_data = np.array(data.T[0])
        yaw_gyr_data = np.array([])
        pitch_gyr_data = np.array([])
        roll_gyr_data = np.array([])
        time = np.linspace(0, len(v_acc_data) / int(samp_freq),
                           len(v_acc_data))
        return IMUData(activity_code, activity_description, v_acc_data,
                       ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                       roll_gyr_data, time)

    def read_UIUC_gaitspeed_dataset(self, path):
        """
        In the GaitSpeedValidation folder, you will find a python script for
        reading and plotting within subject 101's data folder (e.g. B1_T1),
        and the full wearable data for young and older adults in our experiment.
        All subjects but 102 and 223 should have full data, consisting of two
        blocks (B1 and B2) and two trials (T1 and T2) all with gait data performed
        at the subject's comfortable walking pace. In addition, we have a script
        used to convert the .wav binary file into a .csv for further analysis.

        Within a data folder, you should find an info.json file, data folder,
        and a statistics.csv summary file.

        Within main folder, there is rawdataanalysis folder with a few scripts
        needed for raw data reading and conversion.

        Acceleration:
        Units: g
        Frequency: 64
        Unit (bin download): g/256
        :param path:
        :return:
        """
        # Need to create a dataset builder for this
        # Not sure if there is a user demographics anywhere in the data folder
        pass


def main():
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\UIUC_gaitspeed\bin_data\subj_files'
    clinical_demo_path = 'N/A'
    segment_dataset = False
    epoch_size = 0.0
    db = DatasetBuilder()
    uiuc_gait_dataset = db.build_dataset(path, clinical_demo_path, segment_dataset, epoch_size)
    print('\n\n\n')
    v_acc_means = [abs(np.mean(
        user_data.get_imu_data()[IMUDataFilterType.RAW].get_acc_axis_data(
            'vertical'))) for user_data in uiuc_gait_dataset.get_dataset()]
    ml_acc_means = [abs(np.mean(
        user_data.get_imu_data()[IMUDataFilterType.RAW].get_acc_axis_data(
            'mediolateral'))) for user_data in uiuc_gait_dataset.get_dataset()]
    ap_acc_means = [abs(np.mean(
        user_data.get_imu_data()[IMUDataFilterType.RAW].get_acc_axis_data(
            'anteroposterior'))) for user_data in uiuc_gait_dataset.get_dataset()]
    v_count = 0
    ml_count = 0
    ap_count = 0
    for v, ml, ap in zip(v_acc_means, ml_acc_means, ap_acc_means):
        if (7.0 < v < 13.0):
            v_count += 1
        elif (7.0 < ml < 13.0):
            ml_count += 1
        elif (7.0 < ap < 13.0):
            ap_count += 1
    print(v_count)
    print(ml_count)
    print(ap_count)
    print(v_count + ml_count + ap_count)

    # v_acc = uiuc_gait_dataset.get_dataset()[0].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('vertical')
    # ml_acc = uiuc_gait_dataset.get_dataset()[0].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('mediolateral')
    # ap_acc = uiuc_gait_dataset.get_dataset()[0].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('anteroposterior')
    # time = uiuc_gait_dataset.get_dataset()[0].get_imu_data(
    #     IMUDataFilterType.RAW).get_time()
    # plt.plot(time, v_acc, 'r')
    # plt.plot(time, ml_acc, 'g')
    # plt.plot(time, ap_acc, 'b')
    # plt.show()
    # v_acc = uiuc_gait_dataset.get_dataset()[100].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('vertical')
    # ml_acc = uiuc_gait_dataset.get_dataset()[100].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('mediolateral')
    # ap_acc = uiuc_gait_dataset.get_dataset()[100].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('anteroposterior')
    # time = uiuc_gait_dataset.get_dataset()[100].get_imu_data(
    #     IMUDataFilterType.RAW).get_time()
    # plt.plot(time, v_acc, 'r')
    # plt.plot(time, ml_acc, 'g')
    # plt.plot(time, ap_acc, 'b')
    # plt.show()
    # v_acc = uiuc_gait_dataset.get_dataset()[200].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('vertical')
    # ml_acc = uiuc_gait_dataset.get_dataset()[200].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('mediolateral')
    # ap_acc = uiuc_gait_dataset.get_dataset()[200].get_imu_data(
    #     IMUDataFilterType.RAW).get_acc_axis_data('anteroposterior')
    # time = uiuc_gait_dataset.get_dataset()[200].get_imu_data(
    #     IMUDataFilterType.RAW).get_time()
    # plt.plot(time, v_acc, 'r')
    # plt.plot(time, ml_acc, 'g')
    # plt.plot(time, ap_acc, 'b')
    # plt.show()
    print('a')


if __name__ == '__main__':
    main()
