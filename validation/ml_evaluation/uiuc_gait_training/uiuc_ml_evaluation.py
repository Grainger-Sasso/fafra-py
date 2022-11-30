
from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder


class MetricGenerator:
    def __init__(self):
        pass

    def load_dataset(self):
        db = DatasetBuilder()
        dataset_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/bin_data/subj_files/'
        clinical_demo_path = 'N/A'
        segment_dataset = False
        epoch_size = 0.0
        dataset = db.build_dataset(dataset_path, clinical_demo_path,
                                   segment_dataset, epoch_size)

    def generate_input_metrics(self):
        pass


def main():
    mg = MetricGenerator()
    mg.load_dataset()


if __name__ == "__main__":
    main()
