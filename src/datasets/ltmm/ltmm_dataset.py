import os
import glob
import numpy as np
import wfdb
import time
from typing import List, Dict, Tuple
from src.dataset_tools.params.motion_data import MotionData
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.dataset_tools.params.sensor import Sensor
from src.dataset_tools.params.subject import Subject
from src.dataset_tools.params.activity import Activity
from src.dataset_tools.motion_data.acceleration.linear_acceleration.linear_acceleration import LinearAcceleration
from src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
from src.dataset_tools.motion_data.acceleration.angular_acceleration.angular_acceleration import AngularAcceleration
from src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration


class LTMMDataset:
    def __init__(self, dataset_name, dataset_path, clinical_demo_path, report_home_75h_path):
        # Path the the xlsx file with clinical demographic information
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.report_home_75h_path = report_home_75h_path
        self.dataset: Dict[str, LTMMData] = dict()
        self.header_and_data_file_paths = dict()

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_clinical_demo_path(self):
        return self.clinical_demo_path

    def get_report_home_75h_path(self):
        return self.report_home_75h_path

    def get_dataset(self) -> Dict[str, 'LTMMData']:
        return self.dataset

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def read_dataset(self):
        for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
            data = self.build_ltmm_data(name, header_and_data_file_path)
            data.read_data_file()
            data.read_header_file()
            data.set_data_to_float_16()
            self.dataset[name] = data

    def build_ltmm_data(self, name: str, header_and_data_file_paths: Dict[str, str]) -> 'LTMMData':
        data_file_path = header_and_data_file_paths['data_file_path']
        header_file_path = header_and_data_file_paths['header_file_path']
        return LTMMData(name, data_file_path, header_file_path)

    def generate_header_and_data_file_paths(self):
        data_file_paths = {}
        header_file_paths = {}
        # Get all data file paths
        for data_file_path in glob.glob(os.path.join(self.dataset_path, '*.dat')):
            data_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
            data_file_paths[data_file_name] = data_file_path
        # Get all header file paths
        for header_file_path in glob.glob(os.path.join(self.dataset_path, '*.hea')):
            header_file_name = os.path.splitext(os.path.basename(header_file_path))[0]
            header_file_paths[header_file_name] = header_file_path
        # Match corresponding data and header files
        for name, path in data_file_paths.items():
            corresponding_header_file_path = header_file_paths[name]
            self.header_and_data_file_paths[name] = {'data_file_path': path,
                                                'header_file_path': corresponding_header_file_path}


class LTMMData:
    def __init__(self, name, data_file_path, header_file_path):
        self.name = name
        self.data_file_path = data_file_path
        self.header_file_path = header_file_path
        self.data: wfdb.io.record = None
        self.header_data: wfdb.io.record = None

    def set_data_to_float_16(self):
        self.data = np.float16(self.data)

    def get_data(self):
        return self.data

    def get_header_data(self):
        return self.header_data

    def read_data_file(self):
        data_path = os.path.splitext(self.data_file_path)[0]
        self.data = wfdb.rdrecord(data_path)

    def read_header_file(self):
        header_path = os.path.splitext(self.header_file_path)[0]
        self.header_data = wfdb.rdheader(header_path)


class LTMMDatasetConverter:

    def __init__(self, ltmm_dataset):
        self.ltmm_dataset: LTMMDataset = ltmm_dataset

    def convert_ltmm_dataset(self):
        dataset_name = self.ltmm_dataset
        file_format = ''
        activity_ids = ''
        subject_data = ''
        sampling_rate = ''
        sensor_name = ''
        sensor = Sensor('', '', '', '', '')
        sensor_data = {sensor_name: sensor}
        motion_data = self.convert_ltmm_data(self.ltmm_dataset.get_dataset())
        motion_dataset = MotionDataset(dataset_name, file_format, activity_ids,
                                       subject_data, sampling_rate, sensor_data, motion_data)

    def convert_ltmm_data(self, ltmm_data: Dict[str, LTMMData]) -> List[MotionData]:
        motion_data: List[MotionData] = []
        return motion_data


def main():
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset = LTMMDataset(ltmm_dataset_path)
    ltmm_dataset.generate_header_and_data_files_paths()
    time0 = time.time()
    data_CO001 = ltmm_dataset.read_single_data_file('CO001')
    print(time.time()-time0)
    data = data_CO001.get_data()
    header_data = data_CO001.get_header_data()
    print(header_data)
    # wfdb.plot_wfdb(data_CO001.get_data(), title='...')

if __name__ == '__main__':
    main()
