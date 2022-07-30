import glob
import os
import wfdb
import numpy as np

from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


DATASET_NAME = DatasetNames.LTMM


class DatasetBuilder(DatasetBuilder):
    def __init__(self, ):
        super().__init__(DATASET_NAME)
        self.header_and_data_file_paths = dict()
        self.sampling_frequency = 100.0
        # Original units: g,g,g,°/s,°/s,°/s
        # Converted to: m/s^2,m/s^2,m/s^2,°/s,°/s,°/s
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        # Mock height in meters
        self.height = 1.75

    def get_header_and_data_file_paths(self):
        return self.header_and_data_file_paths

    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size):
        self._generate_header_and_data_file_paths(dataset_path)
        dataset = []
        for name, header_and_data_file_path in self.get_header_and_data_file_paths().items():
            data_file_path = header_and_data_file_path['data_file_path']
            header_file_path = header_and_data_file_path['header_file_path']
            data_path = os.path.splitext(data_file_path)[0]
            header_path = os.path.splitext(header_file_path)[0]
            wfdb_record = wfdb.rdrecord(data_path)
            id = wfdb_record.record_name
            print(id)
            data = np.array(wfdb_record.p_signal)
            data = np.float16(data)
            # Convert acceleration data from g to m/s^2
            data[:, 0:3] = data[:, 0:3] * 9.80665
            header_data = wfdb.rdheader(header_path)
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
            imu_data_file_name: str = os.path.split(os.path.splitext(imu_data_file_path)[0])[1]
            imu_metadata_file_path: str = header_file_path
            imu_metadata = IMUMetadata(header_data, self.sampling_frequency, self.units)
            trial = ''
            clinical_demo_data = ClinicalDemographicData(id, age, sex, faller_status, self.height, trial)
            if segment_dataset:
                #TODO: track the segmented data with a linked list
                # Segment the data and build a UserData object for each epoch
                data_segments = self.segment_data(data.T, epoch_size, self.sampling_frequency)
                for segment in data_segments:
                    imu_data = self._generate_imu_data_instance(segment.T, self.sampling_frequency)
                    dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, clinical_demo_path,
                                            {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
            else:
                # Build a UserData object for the whole data
                imu_data = self._generate_imu_data_instance(data, self.sampling_frequency)
                dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, clinical_demo_path,
                                        {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        return Dataset(self.get_dataset_name(), dataset_path, clinical_demo_path, dataset, {})

    def _generate_imu_data_instance(self, data, sampling_freq):
        activity_code = ''
        activity_description = ''
        v_acc_data = np.array(data.T[0])
        ml_acc_data = np.array(data.T[1])
        ap_acc_data = np.array(data.T[2])
        yaw_gyr_data = np.array(data.T[3])
        pitch_gyr_data = np.array(data.T[4])
        roll_gyr_data = np.array(data.T[5])
        time = np.linspace(0, len(v_acc_data) / int(sampling_freq),
                           len(v_acc_data))
        return IMUData(activity_code, activity_description, v_acc_data,
                       ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                       roll_gyr_data, time)

    def _generate_header_and_data_file_paths(self, dataset_path):
        data_file_paths = {}
        header_file_paths = {}
        # Get all data file paths
        for data_file_path in glob.glob(os.path.join(dataset_path, '*.dat')):
            data_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
            data_file_paths[data_file_name] = data_file_path
        # Get all header file paths
        for header_file_path in glob.glob(os.path.join(dataset_path, '*.hea')):
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


def main():
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    segment = True
    epoch_size = 8.0
    db = DatasetBuilder()
    dataset = db.build_dataset(ltmm_dataset_path, clinical_demo_path,
                     segment, epoch_size)
    print(dataset)

if __name__ == '__main__':
    main()
