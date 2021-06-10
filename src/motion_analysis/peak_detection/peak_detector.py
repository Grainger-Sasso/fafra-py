import numpy as np
from scipy import signal


class PeakDetector:
    def __init__(self):
        pass

    def detect_peaks(self, x):
        """Returns peak indices"""
        return signal.find_peaks(x)[0]

    def get_peak_locations(self, x, peak_ixs):
        x = np.array(x)
        return x[peak_ixs]

    def get_largest_peak_ix(self, y, peak_ixs):
        return [loc for _, loc in sorted(zip(np.array(y)[peak_ixs], peak_ixs), reverse=True)][0]
