from typing import Dict
from fafra_py_legacy.src.dataset_tools.params.motion_dataset import MotionDataset
from fafra_py_legacy.src.datasets.sisfall.sisfall_dataset import SisFallDataset
from fafra_py_legacy.src.motion_analysis.filters.motion_filters import MotionFilters
from fafra_py_legacy.src.visualization_tools.motion_visualizer import MotionVisualizer
from fafra_py_legacy.src.motion_analysis.feature_extraction.fall_detection.fall_detector import FallDetector


class FaFRA:

    def __init__(self):
        self.datasets: Dict[str, MotionDataset] = {'SisFall': SisFallDataset(r'C:\Users\gsass\Desktop\Fall Project Master\fafra_py\Fall Datasets\SisFall_csv\SisFall_small_dataset_csv', 'csv')}
        self.motion_visualizer: MotionVisualizer = MotionVisualizer()
        self.motion_filters: MotionFilters = MotionFilters()
        self.fall_detector: FallDetector = FallDetector()

    def read_datasets(self):
        for name, dataset in self.datasets.items():
            dataset.read_dataset()

    def plot_motion_data(self, dataset, subject, activity, trial):
        self.motion_visualizer.plot_motion_data(dataset, subject, activity, trial)

    def detect_fall_in_dataset(self, dataset, method):
        self.fall_detector.detect_falls(dataset, method)




def main():
    fafra = FaFRA()
    fafra.read_datasets()
    dataset_name = 'SisFall'
    dataset = fafra.datasets[dataset_name]
    fafra.detect_fall_in_dataset(dataset, 'sucerquia')
    fafra.plot_motion_data(dataset, 'SA01', 'F05', 'R01')
    # fafra.datasets['SisFall'].write_dataset_to_csv(r'C:\Users\gsass_000\Documents\Fall Project Master\fafra_py_legacy\Fall Datasets\SisFall_csv')
    # fafra.motion_visualizer.plot_motion_data(fafra.datasets[0].motion_data[0])


if __name__ == '__main__':
    main()
