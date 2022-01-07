import os
import glob
import pandas as pd
import numpy as np
from scipy.io import wavfile

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
                      'yaw': None, 'pitch': None, 'roll': None}
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
        # dataset = []
        # for subj_id, data_file_paths in data_file_paths.items():
        #     print(subj_id)
        #     for file_path in data_file_paths:
        #         data = self._read_data_file(file_path)
        #         # Convert accelerometer data from g to m/s^2
        #         data[:, 0:3] = data[:, 0:3] * 9.81
        #         imu_data_file_path: str = file_path
        #         imu_metadata_file_path: str = 'N/A'
        #         clinical_demo_path: str = 'N/A'
        #         imu_metadata = IMUMetadata(None,
        #                                    self.sampling_frequency, self.units)
        #         subj_clin_data = self._get_subj_clin_data(subj_id)
        #         # Build dataset objects from the data and metadata
        #         if segment_dataset:
        #             # TODO: track the segmented data with a linked list
        #             # Segment the data and build a UserData object for each epoch
        #             data_segments = self.segment_data(data, epoch_size,
        #                                               self.sampling_frequency)
        #             for segment in data_segments:
        #                 imu_data = self._generate_imu_data_instance(file_path, segment,
        #                                                             self.sampling_frequency)
        #                 dataset.append(UserData(imu_data_file_path,
        #                                         imu_metadata_file_path,
        #                                         clinical_demo_path,
        #                                         {
        #                                             IMUDataFilterType.RAW: imu_data},
        #                                         imu_metadata,
        #                                         subj_clin_data))
        #         else:
        #             # Build a UserData object for the whole data
        #             imu_data = self._generate_imu_data_instance(file_path, data,
        #                                                         self.sampling_frequency)
        #             dataset.append(
        #                 UserData(imu_data_file_path, imu_metadata_file_path,
        #                          clinical_demo_path,
        #                          {IMUDataFilterType.RAW: imu_data},
        #                          imu_metadata, subj_clin_data))
        # return Dataset(self.get_dataset_name(), dataset_path, clinical_demo_path, dataset, self.activity_codes)

    def _generate_data_file_paths(self, dataset_path):
        data_file_paths = {}
        # Iterate through all of the files in the CSV directory, get all filenames
        subj_ids = next(os.walk(dataset_path))[1]
        subj_paths = [os.path.join(dataset_path, subj_id) for subj_id in subj_ids]
        trial_ids = next(os.walk(subj_paths))[1]
        trial_paths = [os.path.join(dataset_path, trial) for trial in trial_ids]
        for subj_id, trial, path in zip(subj_ids, trial_ids, trial_paths):
            print('a')
            data_file_paths[subj_id] = []
            subj_data_folder = os.path.join(dataset_path, subject_id)
            for data_file_path in glob.glob(
                os.path.join(subj_data_folder, '*.csv')):
                data_file_paths[subject_id].append(data_file_path)
        return data_file_paths

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

    def read_wav(self):
        pass


def main():
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\UIUC_gaitspeed\bin_data\subj_files'
    clinical_demo_path = ''
    segment_dataset = False
    epoch_size = 0.0
    db = DatasetBuilder()
    db.build_dataset(path, clinical_demo_path, segment_dataset, epoch_size)


if __name__ == '__main__':
    main()
