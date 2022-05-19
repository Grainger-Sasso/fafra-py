from src.fibion_mvp.fibion_dataset_builder import FibionDatasetBuilder


class FibionFaFRA:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(self.dataset_path)

    def load_dataset(self, dataset_path):
        dataset_builder = FibionDatasetBuilder()





