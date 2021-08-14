from abc import ABC, abstractmethod

from src.dataset_tools.dataset import Dataset


class DatasetBuilder(ABC):
    def __init__(self, dataset_name, dataset_path, clinical_demo_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_clinical_demo_path(self):
        return self.clinical_demo_path

    @abstractmethod
    def build_dataset(self) -> Dataset:
        pass

