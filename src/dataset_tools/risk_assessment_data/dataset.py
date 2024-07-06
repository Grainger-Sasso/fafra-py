from abc import ABC
from typing import List, Dict

from src.dataset_tools.risk_assessment_data.user_data import UserData


class Dataset(ABC):
    def __init__(self, dataset_name, dataset_path: List['str'],
                 clinical_demo_path: List['str'], dataset: List[UserData],
                 activity_codes: Dict):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.dataset: List[UserData] = dataset
        self.activity_codes: Dict = activity_codes

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_clinical_demo_path(self):
        return self.clinical_demo_path

    def get_dataset(self) -> List['UserData']:
        return self.dataset

    def get_activity_codes(self) -> Dict:
        return self.activity_codes

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_data_by_faller_status(self, faller_status):
        return [user_data for user_data in self.get_dataset() if
                user_data.get_clinical_demo_data().get_faller_status() == faller_status]
