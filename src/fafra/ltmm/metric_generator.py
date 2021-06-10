import numpy as np
import importlib
import os
from typing import List, Tuple
import glob

from definitions import ROOT_DIR
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.datasets.ltmm.ltmm_dataset import LTMMData
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.risk_classification.input_metrics.metric_names import MetricNames


class MetricGenerator:
    def __init__(self):
        self.fft = FastFourierTransform()
        self.peak_detector = PeakDetector()
        self.motion_filters = MotionFilters()

    def generate_metrics(self, ltmm_dataset: List[LTMMData], input_metric_names: Tuple[MetricNames]):
        # Check metric names input by user are all members of metric names enum
        self._check_metric_names_valid(input_metric_names)
        # Initialize intermediate variable for dataset risk classification metrics
        faller_status = []
        dataset_metrics = []
        # Derive metrics for all dataset
        for ltmm_data in ltmm_dataset:
            faller_status.append(int(ltmm_data.get_faller_status()))
            dataset_metrics.append(self._derive_metrics(ltmm_data, input_metric_names))
        return list(dataset_metrics), list(faller_status)

    def _check_metric_names_valid(self, metric_names: Tuple[MetricNames]):
        invalid_metrics = [met for met in metric_names if met not in MetricNames]
        if invalid_metrics:
            raise ValueError(f'The following metrics are not valid metrics: {[met.get_name() for met in invalid_metrics]}')

    def _derive_metrics(self, ltmm_data, metric_names: Tuple[MetricNames]):
        # Initialize the output
        risk_metrics = []
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_frequency = ltmm_data.get_sampling_frequency()
        # Instantiate metric modules for all metric module paths
        metric_modules = [importlib.import_module(module_path).Metric() for
                          module_path in self._generate_metric_module_paths()]
        # Retain only the metric modules selected by metric names
        select_metric_modules = [mod for mod in metric_modules if mod.get_metric_name() in metric_names]
        for mod in select_metric_modules:
            metric = mod.generate_metric(data=v_axis_data, sampling_frequency=sampling_frequency)
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
