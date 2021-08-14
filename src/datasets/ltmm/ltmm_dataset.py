import os
import glob
import numpy as np
import wfdb
import time
import copy
import numpy as np
from typing import List, Dict, Tuple

from src.dataset_tools.dataset import Dataset
from src.dataset_tools.user_data import UserData
from src.dataset_tools.imu_data import IMUData
from src.dataset_tools.imu_metadata import IMUMetadata
from src.dataset_tools.clinical_demographic_data import ClinicalDemographicData
# from src.dataset_tools.params.motion_data import MotionData
# from src.dataset_tools.params.motion_dataset import MotionDataset
# from src.dataset_tools.params.sensor import Sensor
# from src.dataset_tools.params.subject import Subject
# from src.dataset_tools.params.activity import Activity
# from src.dataset_tools.motion_data.acceleration.linear_acceleration.linear_acceleration import LinearAcceleration
# from src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
# from src.dataset_tools.motion_data.acceleration.angular_acceleration.angular_acceleration import AngularAcceleration
# from src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration


class LTMMDataset(Dataset):
    """
    https://physionet.org/content/ltmm/1.0.0/
    """
    def __init__(self, dataset_name, dataset_path, clinical_demo_path, report_home_75h_path):
        super().__init__(dataset_name, dataset_path, clinical_demo_path)
        self.report_home_75h_path = report_home_75h_path
        self.header_and_data_file_paths = dict()
        self.generate_header_and_data_file_paths()
        self.build_dataset()

    def get_report_home_75h_path(self):
        return self.report_home_75h_path

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def get_ltmm_data_by_faller_status(self, faller_status):
        return [user_data for user_data in self.get_dataset() if
                user_data.get_clinical_demo_data().get_faller_status() == faller_status]

    def build_dataset(self):
        imu_data_file_path: str = ''
        imu_metadata_file_path: str = ''
        clinical_demo_file_path: str = ''
        imu_data: IMUData = None
        imu_metadata: IMUMetadata = None
        clinical_demo_data: ClinicalDemographicData = None
        for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
            ltmm_data = self._build_ltmm_data(header_and_data_file_path)
            imu
            ltmm_data.read_data_file()
            ltmm_data.read_header_file()
            ltmm_data.set_data_to_float_16()
            self.dataset.append(ltmm_data)

    def generate_header_and_data_file_paths(self):
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

    def segment_dataset(self, epoch_size):
        segmented_dataset = []
        for data in self.get_dataset():
            data.segment_data(epoch_size)
            new_ltmm_data = []
            for data_seg in data.get_data_segments():
                new_data = copy.deepcopy(data)
                new_data.set_data(data_seg)
                new_data.set_data_segments([])
                new_ltmm_data.append(new_data)
            segmented_dataset.extend(new_ltmm_data)
        self.set_dataset(segmented_dataset)

    def _build_ltmm_data(self, header_and_data_file_paths: Dict[str, str]) -> 'LTMMData':
        data_file_path = header_and_data_file_paths['data_file_path']
        header_file_path = header_and_data_file_paths['header_file_path']
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
        return LTMMData(data_file_path, header_file_path)


class LTMMData(UserData):
    def __init__(self, imu_data_file_path, imu_metadata_file_path, clinical_demo_file_path,
                 imu_data, imu_metadata, clinical_demo_data):
        super().__init__(imu_data_file_path, imu_metadata_file_path, clinical_demo_file_path,
                 imu_data, imu_metadata, clinical_demo_data)

    def set_data_to_float_16(self):
        self.data = np.float16(self.data)

    def read_data_file(self):
        data_path = os.path.splitext(self.data_file_path)[0]
        wfdb_record = wfdb.rdrecord(data_path)
        self.id = wfdb_record.record_name
        self.data = np.array(wfdb_record.p_signal)
        if wfdb_record.comments[0][4:]:
            self.age = float(wfdb_record.comments[0][4:])
        self.sex = wfdb_record.comments[1][4:]
        self._set_faller_status()

    def read_metadata_file(self):
        header_path = os.path.splitext(self.header_file_path)[0]
        self.header_data = wfdb.rdheader(header_path)

    def read_clinical_demo_data_file(self):
        pass

    def segment_data(self, epoch_size):
        """
        Segments data into epochs of a given duration starting from the beginning of the data
        :param epoch_size: duration of epoch to segment data (in seconds)
        :return: data segments of given epoch duration
        """
        total_time = len(self.data.T[0])/self.sampling_frequency
        # Calculate number of segments from given epoch size
        num_of_segs = int(total_time / epoch_size)
        # Check to see if data can be segmented at least one segment of given epoch size
        if num_of_segs > 0:
            self.data_segments = []
            # Counter for the number of segments to be created
            segment_count = range(0, num_of_segs+1)
            # Create segmentation indices
            seg_ixs = [int(seg * self.sampling_frequency * epoch_size) for seg in segment_count]
            for seg_num in segment_count:
                if seg_num != segment_count[-1]:
                    self.data_segments.append(self.data[:][seg_ixs[seg_num]: seg_ixs[seg_num+1]])
                else:
                    continue
        else:
            raise ValueError(f'Data of total time {str(total_time)}s can not be '
                             f'segmented with given epoch size {str(epoch_size)}s')

    def _set_faller_status(self):
        if self.get_id().casefold()[0] == 'f':
            self.faller_status = True
        elif self.get_id().casefold()[0] == 'c':
            self.faller_status = False
        else:
            raise ValueError('LTMM Data faller status unclear from id')


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
