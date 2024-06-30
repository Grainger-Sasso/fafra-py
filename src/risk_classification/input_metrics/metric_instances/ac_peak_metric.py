import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.frequency_analysis.auto_correlation import AutoCorrelation
from src.motion_analysis.peak_detection.peak_detector import PeakDetector


METRIC_NAME = MetricNames.AUTOCORRELATION
METRIC_DATA_TYPE = MetricDataTypes.VERTICAL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        return self._find_largest_ac_peak(kwargs['data'], kwargs['sampling_frequency'])

    def _find_largest_ac_peak(self, data, sampling_frequency):
        """
        autocorrelation peak
        :param data:
        :param sampling_frequency:
        :return:
        """
        ac = AutoCorrelation()
        motion_filters = MotionFilters()
        x_ac, y_ac = ac.autocorrelate(data)
        # Get the ac data for the physiologically relevant freqs
        # x_ac, y_ac = self._get_data_range(x_ac, y_ac)
        # Apply smoothing to ac data
        y_ac_f = motion_filters.apply_lpass_filter(y_ac, 2, sampling_frequency,
                                                   'high')
        # Find largest x and y fft peaks
        largest_ac_peak = self._find_largest_peak(x_ac, y_ac_f)
        return largest_ac_peak

    def _find_largest_peak(self, x, y):
        peak_detector = PeakDetector()
        peak_ixs = peak_detector.detect_peaks(y)[0]
        # Find the largest peak given there are peaks detected, otherwise get the max value from the FFT data
        if len(peak_ixs) > 0:
            max_peak_ix = peak_detector.get_largest_peak_ix(y, peak_ixs)
        else:
            max_peak_ix = np.argmax(y, axis=0)
        max_peak_x_value = int(x[max_peak_ix])
        max_peak_y_value = y[max_peak_ix]
        return max_peak_x_value
        # return [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
        # return [1.0, 1.0]

    def _get_data_range(self, x, y, lower_bd=1.0, upper_bd=3.0):
        """
        Get range of a data based on a mask set on a lower an upper bound of ]
        (Set to physiological range of walking frequency)
        :param x:
        :param y:
        :param lower_bd:
        :param upper_bd:
        :return:
        """
        phys_bds_mask = (lower_bd <= x) & (x <= upper_bd)
        return x[phys_bds_mask], y[phys_bds_mask]
