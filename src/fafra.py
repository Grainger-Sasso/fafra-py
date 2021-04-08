from typing import Dict
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.datasets.sisfall.sisfall_dataset import SisFallDataset
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.visualization_tools.motion_visualizer import MotionVisualizer
from src.motion_analysis.feature_extraction.fall_detection.sucerquia_fall_detector import SucerquiaFallDetector


class FaFRA:

    def __init__(self):
        # Sisfall dataset locations
        # self.datasets: Dict[str, MotionDataset] = {'SisFall': SisFallDataset(r'C:\Users\gsass\Desktop\Fall Project Master\fafra_py\Fall Datasets\SisFall_csv\SisFall_small_dataset_csv', 'csv')}
        self.datasets: Dict[str, MotionDataset] = {'SisFall': SisFallDataset(r'C:\Users\gsass\Desktop\Fall Project Master\fafra_py\Fall Datasets\SisFall_csv\SisFall_dataset_csv', 'csv')}
        # self.datasets: Dict[str, MotionDataset] = {'SisFall': SisFallDataset(r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_py\Fall Datasets\SisFall_csv\SisFall_small_dataset_csv', 'csv')}
        # self.datasets: Dict[str, MotionDataset] = {'SisFall': SisFallDataset(r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_py\Fall Datasets\SisFall_csv\SisFall_dataset_csv', 'csv')}

        # LTMM dataset location

        self.motion_visualizer: MotionVisualizer = MotionVisualizer()
        self.motion_filters: MotionFilters = MotionFilters()
        self.fall_detector: SucerquiaFallDetector = SucerquiaFallDetector(4.44)

    def set_fall_detector(self, fall_detector: SucerquiaFallDetector):
        self.fall_detector = fall_detector

    def read_datasets(self):
        for name, dataset in self.datasets.items():
            dataset.read_dataset()

    def plot_motion_data(self, dataset, subject, activity, trial):
        self.motion_visualizer.plot_motion_data(dataset, subject, activity, trial)


def main():
    fafra = FaFRA()
    fafra.read_datasets()
    fall_detection_results_dir = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\sucerquia_fall_detector\trial'
    # fall_detection_results_dir = r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_testing\sucerquia_fall_detector\test'
    dataset_name = 'SisFall'
    dataset = fafra.datasets[dataset_name]
    # dataset = copy.deepcopy(fafra.datasets[dataset_name])
    # results_df - "measurements": ds_fall_measurements, "predictions": ds_fall_predictions,"comparison": ds_mp_comparison, "indices": np.array(ds_fall_indices)}
    # results_df = fafra.fall_detector.detect_falls_in_motion_dataset(dataset, True, fall_detection_results_dir)
    metric_thresholds = [4.44]
    for metric_threshold in metric_thresholds:
        fall_detector = SucerquiaFallDetector(metric_threshold)
        fafra.set_fall_detector(fall_detector)
        results_df = fafra.fall_detector.detect_falls_in_motion_dataset(dataset, True, fall_detection_results_dir)
    # dataset.apply_lp_filter()
    # dataset.apply_kalman_filter()
    # fafra.plot_motion_data(dataset, 'SA01', 'F05', 'R01')
    # fafra.datasets['SisFall'].write_dataset_to_csv(r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_py_legacy\Fall Datasets\SisFall_csv')
    # fafra.motion_visualizer.plot_motion_data(fafra.datasets[0].motion_data[0])


if __name__ == '__main__':
    main()
