import os
import glob
import numpy as np
import wfdb
import time
from typing import List, Dict, Tuple
# from src.dataset_tools.params.motion_data import MotionData
# from src.dataset_tools.params.motion_dataset import MotionDataset
# from src.dataset_tools.params.sensor import Sensor
# from src.dataset_tools.params.subject import Subject
# from src.dataset_tools.params.activity import Activity
# from src.dataset_tools.motion_data.acceleration.linear_acceleration.linear_acceleration import LinearAcceleration
# from src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
# from src.dataset_tools.motion_data.acceleration.angular_acceleration.angular_acceleration import AngularAcceleration
# from src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration


class LTMMDataset:
    def __init__(self, dataset_name, dataset_path, clinical_demo_path, report_home_75h_path):
        # Path the the xlsx file with clinical demographic information
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.report_home_75h_path = report_home_75h_path
        self.dataset: List[LTMMData] = []
        self.header_and_data_file_paths = dict()

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_clinical_demo_path(self):
        return self.clinical_demo_path

    def get_report_home_75h_path(self):
        return self.report_home_75h_path

    def get_dataset(self) -> List['LTMMData']:
        return self.dataset

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def read_dataset(self):
        for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
            print(name)
            ltmm_data = self._build_ltmm_data(name, header_and_data_file_path)
            ltmm_data.read_data_file()
            ltmm_data.read_header_file()
            ltmm_data.set_data_to_float_16()
            self.dataset.append(ltmm_data)

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

    def _build_ltmm_data(self, name: str, header_and_data_file_paths: Dict[str, str]) -> 'LTMMData':
        data_file_path = header_and_data_file_paths['data_file_path']
        header_file_path = header_and_data_file_paths['header_file_path']
        return LTMMData(name, data_file_path, header_file_path)


class LTMMData:
    def __init__(self, data_file_path, header_file_path):
        self.data_file_path = data_file_path
        self.header_file_path = header_file_path
        self.data = []
        self.header_data: wfdb.io.record = None
        # ADD IN ADDITIONAL ATTRIBUTES FROM THE RECORDS
        self.name = ''
        self.age = 0.00
        self.sex = ''
        self.sampling_frequency = 100.0
        self.axis = ['vertical-acc', 'medio-lateral-acc', 'anterio-posterior-acc', 'yaw', 'pitch', 'roll']
        self.units = ['g', 'g', 'g', 'deg/s', 'deg/s', 'deg/s']

    def get_data_file_path(self):
        return self.data_file_path

    def get_header_file_path(self):
        return self.header_file_path

    def get_data(self):
        return self.data

    def get_header_data(self):
        return self.header_data

    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

    def get_sex(self):
        return self.sex

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_axis(self):
        return self.axis

    def get_units(self):
        return self.units

    def set_data(self, data):
        self.data = data

    def set_data_to_float_16(self):
        self.data = np.float16(self.data)

    def read_data_file(self):
        data_path = os.path.splitext(self.data_file_path)[0]
        wfdb_record = wfdb.rdrecord(data_path)
        self.name = wfdb_record.record_name
        self.data = wfdb_record.p_signal
        if wfdb_record.comments[0][4:]:
            self.age = float(wfdb_record.comments[0][4:])
        self.sex = wfdb_record.comments[1][4:]

    def read_header_file(self):
        header_path = os.path.splitext(self.header_file_path)[0]
        self.header_data = wfdb.rdheader(header_path)


class ClinicalDemographicData:
    def __init__(self, clinical_demo_path):
        self.clinical_demo_path = clinical_demo_path
        self.clinical_demo_data = ''

    def read_clinical_demo_data(self):
        pass

# class LTMMDatasetConverter:
#
#     def __init__(self, ltmm_dataset):
#         self.ltmm_dataset: LTMMDataset = ltmm_dataset
#
#     def convert_ltmm_dataset(self):
#         dataset_name = self.ltmm_dataset.get_dataset_name()
#         file_format = ''
#         activity_ids = ''
#         subject_data = ''
#         sampling_rate = ''
#         sensor_name = ''
#         sensor = Sensor('', '', '', '', '')
#         sensor_data = {sensor_name: sensor}
#         motion_data = self.convert_ltmm_data(self.ltmm_dataset.get_dataset())
#         motion_dataset = MotionDataset(dataset_name, file_format, activity_ids,
#                                        subject_data, sampling_rate, sensor_data, motion_data)
#
#     def convert_ltmm_data(self, ltmm_data: Dict[str, LTMMData]) -> List[MotionData]:
#         motion_data: List[MotionData] = []
#         for name, single_ltmm_data in ltmm_data.items():
#             subject = name
#             activity = ''
#             trial = ''
#             motion_df = single_ltmm_data.get_data()
#             new_motion_data = MotionData()
#         return motion_data


def main():
    # ltmm_dataset_name = 'LTMM'
    ltmm_dataset_name = 'LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    ltmm_dataset = LTMMDataset(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
    ltmm_dataset.generate_header_and_data_file_paths()
    time0 = time.time()
    ltmm_dataset.read_dataset()
    print(time.time()-time0)


if __name__ == '__main__':
    main()
