from abc import ABC, abstractmethod

from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.dataset_builders.dataset_names import DatasetNames


class DatasetBuilder(ABC):

    def __init__(self, dataset_name):
        self.dataset_name: DatasetNames = dataset_name

    def get_dataset_name(self) -> DatasetNames:
        return self.dataset_name

    @abstractmethod
    def build_dataset(self, dataset_name, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size) -> Dataset:
        pass

