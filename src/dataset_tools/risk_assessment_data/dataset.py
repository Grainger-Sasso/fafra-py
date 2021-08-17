from abc import ABC
from typing import List

from src.dataset_tools.risk_assessment_data.user_data import UserData


class Dataset(ABC):
    def __init__(self, dataset_name, dataset_path: List['str'],
                 clinical_demo_path: List['str'], dataset: List[UserData]):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.dataset: List[UserData] = dataset

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_clinical_demo_path(self):
        return self.clinical_demo_path

    def get_dataset(self) -> List['UserData']:
        return self.user_data

    def set_dataset(self, dataset):
        self.user_data = dataset

    def get_data_by_faller_status(self, faller_status):
        return [user_data for user_data in self.get_dataset() if
                user_data.get_clinical_demo_data().get_faller_status() == faller_status]
