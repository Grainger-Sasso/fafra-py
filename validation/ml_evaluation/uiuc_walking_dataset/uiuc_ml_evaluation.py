from validation.ml_evaluation.uiuc_walking_dataset.uiuc_metric_gen import MetricGenerator


class MLEvaluator:
    def __init__(self):
        pass

    def perform_ml_analysis(self, ds_path, clinic_demo_path, output_path, segment_dataset=False, epoch_size=10.0):
        # Generate input features
        features = MetricGenerator().gen_input_features(
            ds_path, clinic_demo_path, output_path, segment_dataset, epoch_size)
        # Train model and characterize performance
        # Analyze input features
        pass



def main():
    ds_path = ''
    clinic_demo_path = ''
    output_path = ''
    segment_dataset = False
    epoch_size = 10.0
    ml_e = MLEvaluator()
    ml_e.perform_ml_analysis(
        ds_path, clinic_demo_path, output_path, segment_dataset=segment_dataset, epoch_size=epoch_size)


if __name__ == "__main__":
    main()
