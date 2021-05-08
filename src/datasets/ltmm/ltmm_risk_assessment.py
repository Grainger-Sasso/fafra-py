import numpy as np
from typing import List
from enum import Enum
from src.datasets.ltmm.ltmm_dataset import LTMMDataset, LTMMData
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.visualization_tools.motion_visualizer import MotionVisualizer, PlottingData
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.input_metrics.risk_classification_input_metrics import RiskClassificationInputMetrics
from src.visualization_tools.classification_visualizer import ClassificationVisualizer


class LTMMRiskAssessment:
    def __init__(self, ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path):
        self.ltmm_dataset = LTMMDataset(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
        self._initialize_dataset()
        self.motion_filters = MotionFilters()
        self.risk_classifier = SVMRiskClassifier()
        self.rc_viz = ClassificationVisualizer()
        self.input_metrics: List[RiskClassificationInputMetrics] = []
        self.input_metric_names = RiskClassificationMetricNames
        self.fft = FastFourierTransform()
        self.peak_detector = PeakDetector()

    def assess_cohort_risk(self):
        # Filter the data
        self.apply_lp_filter()
        # Segment the datasets into smaller epochs to have a greater number of data points
        self.ltmm_dataset.segment_dataset(10.0)
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Perform feature extraction, two features, peak location of the fft, and raw rms
        faller_metrics = self._derive_input_metrics(ltmm_faller_data)
        non_faller_metrics = self._derive_input_metrics(ltmm_non_faller_data)
        # Pair extracted features with input data outcomes (x, y), format: x - ndarray {100,2} raw metrics; y - ndarray {100,} binary 1,0
        model_training_x, model_training_y = self._format_model_training_data(faller_metrics, non_faller_metrics)
        # Train risk model on features
        self._train_risk_model(model_training_x, model_training_y)
        # Plot the trained risk features
        self._viz_trained_model(model_training_x, model_training_y)
        # Make inference on the cohort
        # Output inferences to csv

    def apply_lp_filter(self):
        for ltmm_data in self.ltmm_dataset.get_dataset():
            lpf_data_all_axis = []
            sampling_rate = ltmm_data.get_sampling_frequency()
            data = ltmm_data.get_data()
            for axis in data.T:
                lpf_data_all_axis.append(self.motion_filters.apply_lpass_filter(axis, sampling_rate))
            lpf_data_all_axis = np.array(lpf_data_all_axis).T
            ltmm_data.set_data(lpf_data_all_axis)

    def _viz_trained_model(self, x, y):
        self.rc_viz.plot_classification(self.risk_classifier.get_model(), x, y)

    def _train_risk_model(self, x, y):
        self.risk_classifier.generate_model()
        self.risk_classifier.fit_model(x, y)

    def _format_model_training_data(self, faller_metrics, non_faller_metrics):
        faller_x = [[metric[self.input_metric_names.fft_peak.value], metric[self.input_metric_names.rms.value]]
                    for metric in faller_metrics]
        faller_y = np.ones(len(faller_metrics))
        non_faller_x = [[metric[self.input_metric_names.fft_peak.value], metric[self.input_metric_names.rms.value]]
                        for metric in non_faller_metrics]
        non_faller_y = np.zeros(len(faller_metrics))
        model_training_x = faller_x + non_faller_x
        model_training_y = faller_y + non_faller_y
        return np.array(model_training_x), np.array(model_training_y)

    def _derive_input_metrics(self, ltmm_dataset):
        # TODO: include the ltmm_data fall_status as an output, done here instead of elsewhere
        # Initialize intermediate variable for dataset risk classification metrics
        dataset_metrics = []
        # Derive metrics for all dataset
        for ltmm_data in ltmm_dataset:
            dataset_metrics.append(self._derive_metrics(ltmm_data))
        norm_metrics = self._normalize_input_metrics(dataset_metrics)
        return norm_metrics

    def _derive_metrics(self, ltmm_data):
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_rate = ltmm_data.get_sampling_frequency()
        # Get largest peak location of walking fft for vertical axis
        x_fft_peak_val, y_fft_peak_val = self._find_largest_fft_peak(v_axis_data, sampling_rate)
        # Get RMS
        rms = self._get_rms(v_axis_data)
        # Get mean
        mean = self._get_mean(v_axis_data)
        # Get standard deviation
        std = self._get_std(v_axis_data)
        return [x_fft_peak_val, y_fft_peak_val, rms, mean, std]

    def _get_mean(self, data):
        return np.mean(data)

    def _get_std(self, data):
        return np.std(data)

    # def _normalize_input_metrics(self, dataset_metrics):
    #     input_metric_names = [metric.value for metric in self.input_metric_names]
    #     for metric_name in input_metric_names:
    #         max_metric_value = max(dataset_metrics, key=lambda x: x[metric_name])[metric_name]
    #         for metric in dataset_metrics:
    #             metric[metric_name] = metric[metric_name] / max_metric_value
    #     return dataset_metrics

    def _get_rms(self, data):
        return self.motion_filters.calculate_rms(data)

    def _find_largest_fft_peak(self, data, sampling_rate):
        x_fft, y_fft = self.fft.perform_fft(data, sampling_rate)
        # Get the fft data for the physiologically relevant freqs
        # x_fft, y_fft = self._get_data_range(x_fft, y_fft)
        # Apply smoothing to fft data
        x_fft = self.motion_filters.apply_lpass_filter(x_fft, sampling_rate)
        y_fft = self.motion_filters.apply_lpass_filter(y_fft, sampling_rate)
        # Find largest x and y fft peaks
        # TODO: Add try/except to remove this data object from the input if no peaks are found in fft data
        return self._find_largest_peak(x_fft, y_fft)

    def _get_data_range(self, x, y, lower_bd=1.0, upper_bd=3.0):
        phys_bds_mask = (lower_bd <= x) & (x <= upper_bd)
        return x[phys_bds_mask], y[phys_bds_mask]

    def _find_largest_peak(self, x, y):
        peak_ixs = self.peak_detector.detect_peaks(y)
        if len(peak_ixs) > 0:
            max_peak_ix = self.peak_detector.get_largest_peak_ix(y, peak_ixs)
            max_peak_x_value = x[max_peak_ix]
            max_peak_y_value = y[max_peak_ix]
            return max_peak_x_value, max_peak_y_value
        else:
            raise ValueError('No peaks found in fft data')


    def _initialize_dataset(self):
        self.ltmm_dataset.generate_header_and_data_file_paths()
        self.ltmm_dataset.read_dataset()

    def _normalize_input_metrics(self, input_metrics):
        return np.apply_along_axis(self.motion_filters.unit_vector_norm, 0, np.array(input_metrics))



class RiskClassificationMetricNames(Enum):
    fft_peak_x_value = 'fft_peak_x_value'
    fft_peak_y_value = 'fft_peak_y_value'
    rms = 'rms'
    mean = 'mean'
    std_dev = 'std_dev'


def main():
    # ltmm_dataset_name = 'LTMM'
    ltmm_dataset_name = 'LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
    # data_before_lpf = ltmm_ra.ltmm_dataset.get_dataset()[1].get_data().T[0,:]
    ltmm_ra.assess_cohort_risk()
    # data_after_lpf = ltmm_ra.ltmm_dataset.get_dataset()[1].get_data().T[0,:]
    # plot_before = PlottingData(data_before_lpf, 'Unfiltered', '')
    # plot_after = PlottingData(data_after_lpf, 'Filtered', '')
    # viz = MotionVisualizer()
    # viz.plot_data([plot_before, plot_after])

if __name__ == '__main__':
    main()
