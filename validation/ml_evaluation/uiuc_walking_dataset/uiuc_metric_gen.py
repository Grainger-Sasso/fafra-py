from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder


class MetricGenerator:
    def __init__(self):
        pass

    def gen_input_features(self, ds_path, clinic_demo_path, output_path, segment_dataset, epoch_size):
        features = []
        # Load dataset
        ds = self.load_dataset(ds_path, clinic_demo_path, segment_dataset, epoch_size)
        ###
        # TODO: Characterize dataset (for both seg and non-seg) number of trials/samples, total time available, etc see notes. Export to a file
        ###
        # Preprocess dataset
        # Generate SKDH
        # Generate custom metrics
        # Format input metrics
        # Export input metrics
        return features

    def load_dataset(self, ds_path, clinic_demo_path, segment_dataset, epoch_size):
        db = DatasetBuilder()
        dataset = db.build_dataset(ds_path, clinic_demo_path,
                                   segment_dataset, epoch_size)
        return dataset


def main():
    mg = MetricGenerator()
    ds_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/bin_data/subj_files/'
    clinic_demo_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/participant_metadata/Data_CHI2021_Carapace.xlsx'
    output_path = ''
    mg.load_dataset(ds_path, clinic_demo_path, output_path)


if __name__ == "__main__":
    main()
