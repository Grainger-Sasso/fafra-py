import numpy as np
from scipy import signal


class PeakDetector:
    def __init__(self):
        pass

    def detect_peaks(self, x, height=None,
                threshold=None, distance=None,prominence=None,
                width=None, wlen=None, rel_height=0.5,
                     plateau_size=None):
        """Returns peak indices"""
        peaks = signal.find_peaks(x, height=height,
                threshold=threshold, distance=distance,prominence=prominence,
                width=width, wlen=wlen, rel_height=rel_height,
                                  plateau_size=plateau_size)
        return peaks

    def get_peak_locations(self, x, peak_ixs):
        x = np.array(x)
        return x[peak_ixs]

    def get_largest_peak_ix(self, y, peak_ixs):
        return [loc for _, loc in sorted(zip(np.array(y)[peak_ixs], peak_ixs), reverse=True)][0]
