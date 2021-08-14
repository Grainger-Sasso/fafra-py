import glob
import os
import wfdb
import numpy as np

from src.dataset_tools.dataset_builder import DatasetBuilder
from src.dataset_tools.dataset import Dataset
from src.dataset_tools.user_data import UserData
from src.dataset_tools.imu_data import IMUData
from src.dataset_tools.imu_metadata import IMUMetadata
from src.dataset_tools.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.clinical_demographic_data import ClinicalDemographicData
from src.datasets.ltmm.ltmm_dataset import LTMMDataset, LTMMData


class LTMMDatasetBuilder(DatasetBuilder):
    def __init__(self, dataset_name, dataset_path, clinical_demo_path, report_home_75h_path):
        super().__init__(dataset_name, dataset_path, clinical_demo_path)
        self.report_home_75h_path = report_home_75h_path
        self.header_and_data_file_paths = dict()

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def build_dataset(self):
        self._generate_header_and_data_file_paths()
        dataset = []
        for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
            data_file_path = header_and_data_file_path['data_file_path']
            header_file_path = header_and_data_file_path['header_file_path']
            data_path = os.path.splitext(data_file_path)[0]
            wfdb_record = wfdb.rdrecord(data_path)
            id = wfdb_record.record_name
            data = np.array(wfdb_record.p_signal)
            if wfdb_record.comments[0][4:]:
                age = float(wfdb_record.comments[0][4:])
            sex = wfdb_record.comments[1][4:]
            if id.casefold()[0] == 'f':
                faller_status = True
            elif id.casefold()[0] == 'c':
                faller_status = False
            else:
                raise ValueError('LTMM Data faller status unclear from id')
            imu_data_file_path: str = data_file_path
            imu_metadata_file_path: str = header_file_path
            clinical_demo_file_path: str = 'N/A'
            v_acc_data = np.array(data.T[0])
            ml_acc_data = np.array(data.T[1])
            ap_acc_data = np.array(data.T[2])
            yaw_gyr_data = np.array(data.T[3])
            pitch_gyr_data = np.array(data.T[4])
            roll_gyr_data = np.array(data.T[5])
            imu_data: IMUData = IMUData(v_acc_data, ml_acc_data, ap_acc_data,
                                        v_gyr_data, ml_gyr_data, ap_gyr_data)
            sampling_freq = 100.0
            units = {'vertical-acc': 'g', 'mediolateral-acc': 'g', 'anteroposterior-acc': 'g',
                     'yaw': 'deg/s', 'pitch': 'deg/s', 'roll': 'deg/s'}
            imu_metadata: IMUMetadata = IMUMetadata(sampling_freq, units)
            # Mock height in cm
            height = 175.0
            clinical_demo_data: ClinicalDemographicData = ClinicalDemographicData(id, age, sex,
                                                                                  faller_status, height)
            dataset.append(UserData(imu_data_file_path, imu_metadata_file_path, clinical_demo_file_path,
                                 {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))

        return LTMMDataset(self.get_dataset_name(), self.get_dataset_path(),
                           self.get_clinical_demo_path(), (self.report_home_75h_path,
                                                           dataset)


    def _generate_header_and_data_file_paths(self):
        data_file_paths = {}
        header_file_paths = {}
        # Get all data file paths
        for data_file_path in glob.glob(os.path.join(self.get_dataset_path(), '*.dat')):
            data_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
            data_file_paths[data_file_name] = data_file_path
        # Get all header file paths
        for header_file_path in glob.glob(os.path.join(self.get_dataset_path(), '*.hea')):
            header_file_name = os.path.splitext(os.path.basename(header_file_path))[0]
            header_file_paths[header_file_name] = header_file_path
        # Match corresponding data and header files
        for name, path in data_file_paths.items():
            corresponding_header_file_path = header_file_paths[name]
            self.header_and_data_file_paths[name] = {'data_file_path': path,
                                                'header_file_path': corresponding_header_file_path}

    def get_axis_acc_data(self, axis):
        if axis == 'vertical':
            data = np.array(self.get_data().T[0])
        elif axis == 'mediolateral':
            data = np.array(self.get_data().T[1])
        elif axis == 'anteroposterior':
            data = np.array(self.get_data().T[2])
        else:
            raise ValueError(f'{axis} is not a valid axis')
        return data

