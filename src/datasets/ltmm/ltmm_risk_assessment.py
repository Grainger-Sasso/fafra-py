import numpy as np
from src.datasets.ltmm.ltmm_dataset import LTMMDataset
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform


class LTMMRiskAssessment:
    def __init__(self, ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path):
        self.ltmm_dataset = LTMMDataset(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
        self.motion_filters = MotionFilters()
        self.fft = FastFourierTransform()

    def assess_cohort_risk(self):
        # Initialize the dataset
        self._initialize_dataset()
        # Filter the data
        self._apply_lp_filter()
        # Perform feature extraction, two features, peak location of the fft, and raw rms
        # Train risk model on features
        # Make inference on the cohort
        # Output inferences to csv
        pass

    def _apply_lp_filter(self):
        lpf_data_all_axis = []
        for ltmm_data in self.ltmm_dataset.get_dataset():
            sampling_rate = ltmm_data.get_sampling_frequency()
            data = ltmm_data.get_data()
            for axis in data.T:
                lpf_data_all_axis.append(self.motion_filters.apply_lpass_filter(axis, sampling_rate))
            lpf_data_all_axis = np.array(lpf_data_all_axis).T
            ltmm_data.set_data(lpf_data_all_axis)
        # TODO test the application of the lpf with plotting

    def _initialize_dataset(self):
        self.ltmm_dataset.generate_header_and_data_file_paths()
        self.ltmm_dataset.read_dataset()

def main():
    # ltmm_dataset_name = 'LTMM'
    ltmm_dataset_name = 'LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
    ltmm_ra.assess_cohort_risk()