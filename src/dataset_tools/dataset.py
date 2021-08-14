from abc import ABC, abstractmethod
from typing import List

from src.dataset_tools.user_data import UserData


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
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset
