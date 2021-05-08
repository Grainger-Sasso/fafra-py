from scipy import signal


class PeakDetector:
    def __init__(self):
        pass

    def detect_peaks(self, x, y, largest_peak=False):
        """Returns peak indices"""
        peak_ix = signal.find_peaks(y)[0]
        if largest_peak:
            peak_ix = [loc for _, loc in sorted(zip(y[peak_ix], x[peak_ix]), reverse=True)][0]
        return peak_ix
