from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.metric_names import MetricNames


class ModelTrainer:
    def __init__(self, dataset_path, clinical_demo_path, segment_dataset, epoch_size , custom_metric_names):
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.segment_dataset = segment_dataset
        self.epoch_size = epoch_size
        self.dataset = self.build_dataset()
        self.custom_metric_names = custom_metric_names

    def generate_model(self, output_path, file_name):
        # Generate custom metrics
        custom_input_metrics: InputMetrics = self.generate_custom_metrics()
        # Generate SKDH metrics
        skdh_input_metrics = self.generate_skdh_metrics()
        # Format input metrics
        input_metrics = self.format_input_metrics(custom_input_metrics, skdh_input_metrics)
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

    def generate_custom_metrics(self) -> InputMetrics:
        mg = MetricGenerator()
        return mg.generate_metrics(
            self.dataset.get_dataset(),
            self.custom_metric_names
        )

    def generate_skdh_metrics(self):
        return 0

    def format_input_metrics(self, custom_input_metrics, skdh_input_metrics):
        pass


def main():
    dp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/'
    cdp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    seg = False
    epoch = 0.0
    metric_names = tuple(
        [
            MetricNames.AUTOCORRELATION,
            MetricNames.FAST_FOURIER_TRANSFORM,
            MetricNames.MEAN,
            MetricNames.ROOT_MEAN_SQUARE,
            MetricNames.STANDARD_DEVIATION,
            MetricNames.SIGNAL_ENERGY,
            MetricNames.COEFFICIENT_OF_VARIANCE,
            MetricNames.ZERO_CROSSING,
            MetricNames.SIGNAL_MAGNITUDE_AREA
        ]
    )
    mt = ModelTrainer(dp, cdp, seg, epoch, metric_names)
    print('yup')


if __name__ == '__main__':
    main()

