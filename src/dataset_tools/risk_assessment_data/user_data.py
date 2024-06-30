import os
from typing import Dict

from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


class UserData:
    def __init__(self, imu_data_file_path, imu_data_file_name, imu_metadata_file_path, clinical_demo_file_path,
                 imu_data, imu_metadata, clinical_demo_data):
        self.imu_data_file_path = imu_data_file_path
        self.imu_data_file_name = imu_data_file_name
        self.imu_metadata_file_path = imu_metadata_file_path
        self.clinical_demo_file_path = clinical_demo_file_path
        self.imu_data: Dict[IMUDataFilterType: IMUData] = imu_data
        self.imu_metadata: IMUMetadata = imu_metadata
        self.clinical_demo_data: ClinicalDemographicData = clinical_demo_data

    def get_imu_data_file_path(self):
        return self.imu_data_file_path

    def get_imu_data_file_name(self):
        return self.imu_data_file_name

    def get_imu_metadata_file_path(self):
        return self.imu_metadata_file_path

    def get_clinical_demo_file_path(self):
        return self.clinical_demo_file_path

    def get_imu_data(self, filter_type: IMUDataFilterType = None):
        if filter_type:
            data = self.imu_data[filter_type]
        else:
            data = self.imu_data
        return data

    def get_imu_metadata(self):
        return self.imu_metadata

    def get_clinical_demo_data(self):
        return self.clinical_demo_data

    def add_filtered_data(self, imu_data: IMUData, filter_type: IMUDataFilterType):
        self.imu_data[filter_type] = imu_data
