import numpy as np
import importlib
import os
from typing import List, Tuple
import glob
from definitions import ROOT_DIR

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric


class MetricGenerator:
    def __init__(self):
        self.fft = FastFourierTransform()
        self.peak_detector = PeakDetector()
        self.motion_filters = MotionFilters()
        self.metric_modules = []

    def get_metric_modules(self):
        return self.metric_modules

    def write_metrics_csv(self, metrics, faller_status_labels, path, filename):
        x_filename = filename + '_x_metrics.csv'
        y_filename = filename + '_y_labels.csv'
        x_full_path = os.path.join(path, x_filename)
        y_full_path = os.path.join(path, y_filename)
        np.savetxt(x_full_path, metrics, delimiter=",")
        np.savetxt(y_full_path, faller_status_labels, delimiter=",")

    def generate_metrics(self, dataset: List[UserData],
                         input_metric_names: Tuple[MetricNames]) -> InputMetrics:
        # Initialize output
        input_metrics = InputMetrics()
        # Check metric names input by user are all members of metric names enum
        self._check_metric_names_valid(input_metric_names)
        # Initialize intermediate variable for dataset risk classification labels
        labels = []
        # Import metric modules
        self._import_metric_modules(input_metric_names)
        # Derive metrics for all dataset
        for mod in self.metric_modules:
            input_metric, label = self._derive_metrics(mod, dataset)
            input_metrics.set_metric(input_metric.get_name(), input_metric)
            labels.append(label)
        # CHECK ALL ELEMENTS OF FALLER STATUS ARE SAME, TAKE FIRST VALUE
        label = self._check_metric_labels(labels)
        input_metrics.set_labels(label)
        return input_metrics

    def _check_metric_labels(self, labels):
        label = labels[0]
        for other_label in labels[1:]:
            if label != other_label:
                raise ValueError('Faller status not equal')
        return label

    def _check_metric_names_valid(self, metric_names: Tuple[MetricNames]):
        invalid_metrics = [met for met in metric_names if met not in MetricNames]
        if invalid_metrics:
            raise ValueError(f'The following metrics are not valid metrics: {[met.get_name() for met in invalid_metrics]}')

    def _import_metric_modules(self, metric_names):
        # Instantiate metric modules for all metric module paths
        metric_modules = [importlib.import_module(module_path).Metric() for
                          module_path in self._generate_metric_module_paths()]
        # Retain only the metric modules selected by metric names
        self.metric_modules = [mod for mod in metric_modules if
                               mod.get_metric_name() in metric_names]

    def _get_faller_status(self, dataset):
        faller_status = []
        for user_data in dataset:
            faller_status.append(
                int(user_data.get_clinical_demo_data().get_faller_status()))
        return faller_status

    def _derive_metrics(self, mod: RiskClassificationInputMetric,
                        dataset: List[UserData]):
        # Initialize the output
        risk_metric = []
        labels = []
        # Derive metric for all user data in dataset
        for user_data in dataset:
            sampling_frequency = user_data.get_imu_metadata().get_sampling_frequency()
            data_type = mod.get_data_type()
            self._check_metric_data_type(data_type)
            data = self._get_metric_data_type(data_type, user_data)
            metric = mod.generate_metric(data=data, sampling_frequency=sampling_frequency)
            if isinstance(metric, list) and all(isinstance(m, float) or isinstance(m, int) for m in metric):
                risk_metric.extend(metric)
            elif isinstance(metric, int) or isinstance(metric, float):
                risk_metric.append(metric)
            labels.append(
                int(user_data.get_clinical_demo_data().get_faller_status()))
            if len(risk_metric) != len(labels):
                raise ValueError('Metrics and labels not of same length')
        return InputMetric(mod.get_metric_name(), np.array(risk_metric)), labels

    def _get_metric_data_type(self, data_type, user_data: UserData):
        if data_type == MetricDataTypes.USER_DATA:
            data = user_data
        elif data_type == MetricDataTypes.VERTICAL:
            data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        elif data_type == MetricDataTypes.RESULTANT:
            tri_data = user_data.get_imu_data()[IMUDataFilterType.LPF].get_triax_acc_data()
            data = self.motion_filters.calculate_resultant_vector(
                tri_data['vertical'],
                tri_data['mediolateral'],
                tri_data['anteroposterior'])
        else:
            raise ValueError(
                f'Data type provided is not recognized {data_type}')
        return data

    def _check_metric_data_type(self, data_type):
        if data_type not in MetricDataTypes:
            raise ValueError(
                f'The following metric data type is invalid: {data_type.get_name()}')

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
