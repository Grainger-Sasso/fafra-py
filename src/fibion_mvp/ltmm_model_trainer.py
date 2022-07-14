from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder
from src.risk_classification.input_metrics.metric_generator import MetricGenerator


class ModelTrainer:
    def __init__(self, dataset_path, clinical_demo_path, segment_dataset, epoch_size , metric_names):
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.segment_dataset = segment_dataset
        self.epoch_size = epoch_size
        self.dataset = self.build_dataset()
        self.metric_names = metric_names

    def generate_model(self, output_path, file_name):
        # Generate custom metrics
        # Generate SKDH metrics
        # Format input metrics
        # Train model on input metrics
        # Export model
        pass

    def build_dataset(self):
        db = DatasetBuilder()
        return db.build_dataset(
            self.dataset_path,
            self.clinical_demo_path,
            self.segment_dataset,
            self.epoch_size
        )


def main():
    dp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/'
    cdp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    seg = False
    epoch = 0.0
    metric_names = []
    mt = ModelTrainer(dp, cdp, seg, epoch, metric_names)
    print('yup')


if __name__ == '__main__':
    main()

