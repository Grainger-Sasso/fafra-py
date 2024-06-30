import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector


METRIC_NAME = MetricNames.FAST_FOURIER_TRANSFORM
METRIC_DATA_TYPE = MetricDataTypes.VERTICAL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        return self._find_largest_fft_peak(kwargs['data'], kwargs['sampling_frequency'])

    def _find_largest_fft_peak(self, data, sampling_frequency):
        fft = FastFourierTransform()
        motion_filters = MotionFilters()
        x_fft, y_fft = fft.perform_fft(data, sampling_frequency)
        # Get the fft data for the physiologically relevant freqs
        # x_fft, y_fft = self._get_data_range(x_fft, y_fft)
        # Apply smoothing to fft data
        # x_fft = motion_filters.apply_lpass_filter(x_fft, sampling_frequency)
        # y_fft_f = motion_filters.apply_lpass_filter(y_fft, 15,
        #                                             sampling_frequency)
        y_fft_f = motion_filters.apply_lpass_filter(y_fft, 6,
                                                    sampling_frequency)
        # Find largest x and y fft peaks
        largest_fft_peak = self._find_largest_peak(x_fft, y_fft_f)
        return largest_fft_peak

    def _find_largest_peak(self, x, y):
        peak_detector = PeakDetector()
        peak_ixs = peak_detector.detect_peaks(y)[0]
        # Find the largest peak given there are peaks detected, otherwise get the max value from the FFT data
        if len(peak_ixs) > 0:
            max_peak_ix = peak_detector.get_largest_peak_ix(y, peak_ixs)
        else:
            max_peak_ix = np.argmax(y, axis=0)
        max_peak_x_value = x[max_peak_ix]
        max_peak_y_value = y[max_peak_ix]
        return max_peak_x_value

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
