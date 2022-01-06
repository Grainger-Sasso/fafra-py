import os
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
    def __init__(self, ):
        # TODO: add second sisfall dataset for the second accelerometer in dataset, currently not being used
        super().__init__(DATASET_NAME)
        self.sampling_frequency = 200.0
        # Original units: g,g,g,°/s,°/s,°/s
        # Converted to: m/s^2,m/s^2,m/s^2,°/s,°/s,°/s
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        # Adults were not screened for fall risk, therefor none of them are assumed to be fallers
        self.subject_data = {}
        self.activity_codes = {}

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        data_file_paths = self._generate_data_file_paths(dataset_path)
        dataset = []
        for subj_id, data_file_paths in data_file_paths.items():
            print(subj_id)
            for file_path in data_file_paths:
                data = self._read_data_file(file_path)
                # Convert accelerometer data from g to m/s^2
                data[:, 0:3] = data[:, 0:3] * 9.81
                imu_data_file_path: str = file_path
                imu_metadata_file_path: str = 'N/A'
                clinical_demo_path: str = 'N/A'
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
                        UserData(imu_data_file_path, imu_metadata_file_path,
                                 clinical_demo_path,
                                 {IMUDataFilterType.RAW: imu_data},
                                 imu_metadata, subj_clin_data))
        return Dataset(self.get_dataset_name(), dataset_path, clinical_demo_path, dataset, self.activity_codes)

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
        path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\validation\gait_speed\test_data\UIUC\GaitSpeedValidation

    def read_wav():
        pass