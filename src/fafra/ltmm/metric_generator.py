import numpy as np
import importlib
import os
from typing import List
import glob

from definitions import ROOT_DIR
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.datasets.ltmm.ltmm_dataset import LTMMDataset, LTMMData
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector


class MetricGenerator:
    def __init__(self):
        self.fft = FastFourierTransform()
        self.peak_detector = PeakDetector()
        self.motion_filters = MotionFilters()

    def generate_metrics(self, ltmm_dataset: List[LTMMData]):
        # Initialize intermediate variable for dataset risk classification metrics
        faller_status = []
        dataset_metrics = []
        # Derive metrics for all dataset
        for ltmm_data in ltmm_dataset:
            faller_status.append(int(ltmm_data.get_faller_status()))
            dataset_metrics.append(self._derive_metrics(ltmm_data))
        norm_metrics = self._normalize_input_metrics(np.array(dataset_metrics))
        return list(norm_metrics), list(faller_status)

    def _normalize_input_metrics(self, input_metrics):
        norm_metrics = np.apply_along_axis(self.motion_filters.unit_vector_norm, 0, np.array(input_metrics))
        return norm_metrics

    def _derive_metrics(self, ltmm_data):

        # Initialize the output
        risk_metrics = []
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_frequency = ltmm_data.get_sampling_frequency()
        # Generate path for all metric modules, iterate through them
        for module_path in self._generate_metric_module_paths():
            # Dynamically load up the metric instances, creating a list of objects
            metric_module = importlib.import_module(module_path)
            # Call the generate metric method for every metric object
            risk_metrics.append(metric_module.Metric().generate_metric(data=v_axis_data, sampling_frequency=sampling_frequency))
        return risk_metrics

    def _generate_metric_module_paths(self):
        module_root = 'src.risk_classification.input_metrics.metric_instances.'
        module_names = glob.glob(os.path.join(ROOT_DIR, 'src', 'risk_classification', 'input_metrics',
                                              'metric_instances','*_metric.py'), recursive=True)
        module_names = [(os.path.splitext(mod_name)[0]) for mod_name in module_names]
        return [module_root + mod_name for mod_name in module_names]


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

    def oldies(self, ltmm_data: LTMMData):
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_rate = ltmm_data.get_sampling_frequency()
