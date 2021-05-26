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

    def _derive_metrics(self, ltmm_data):
        # Initialize the output
        risk_metrics = []
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_frequency = ltmm_data.get_sampling_frequency()
        # Generate path for all metric modules, iterate through them
        for module_path in self._generate_metric_module_paths():
            # Dynamically load up the metric instances, creating a list of objects
            metric_module = importlib.import_module(module_path).Metric()
            metric = metric_module.generate_metric(data=v_axis_data,
                                                   sampling_frequency=sampling_frequency)
            if isinstance(metric, list) and all(isinstance(m, float) or isinstance(m, int) for m in metric):
                risk_metrics.extend(metric)
            elif isinstance(metric, int) or isinstance(metric, float):
                risk_metrics.append(metric)
        return risk_metrics

    def _generate_metric_module_paths(self):
        module_root = 'src.risk_classification.input_metrics.metric_instances.'
        module_names = glob.glob(os.path.join(ROOT_DIR, 'src', 'risk_classification', 'input_metrics',
                                              'metric_instances', '*_metric.py'), recursive=True)
        module_names = [os.path.splitext(os.path.basename(mod_name))[0] for mod_name in module_names]
        module_names = [module_root + mod_name for mod_name in module_names]
        return module_names

    def _normalize_input_metrics(self, input_metrics):
        norm_metrics = np.apply_along_axis(self.motion_filters.unit_vector_norm, 0, np.array(input_metrics))
        return norm_metrics
