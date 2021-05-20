import numpy as np

from src.risk_classification.input_metrics.risk_classification_input_metrics import RiskClassificationInputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.datasets.ltmm.ltmm_dataset import LTMMData
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.peak_detection.peak_detector import PeakDetector


class FFTPeakMetric(RiskClassificationInputMetric):
    def __init__(self):
        self.fft = FastFourierTransform()
        self.motion_filters = MotionFilters()
        self.peak_detector = PeakDetector()

    def generate_metric(self, ltmm_data: LTMMData):
        v_axis_data = np.array(ltmm_data.get_data().T[0])
        sampling_rate = ltmm_data.get_sampling_frequency()
        return self._find_largest_fft_peak(v_axis_data, sampling_rate)

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

    def _find_largest_peak(self, x, y):
        peak_ixs = self.peak_detector.detect_peaks(y)
        if len(peak_ixs) > 0:
            max_peak_ix = self.peak_detector.get_largest_peak_ix(y, peak_ixs)
            max_peak_x_value = x[max_peak_ix]
            max_peak_y_value = y[max_peak_ix]
            return max_peak_x_value, max_peak_y_value
        else:
            raise ValueError('No peaks found in fft data')
