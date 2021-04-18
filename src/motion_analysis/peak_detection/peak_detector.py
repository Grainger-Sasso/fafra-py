from scipy import signal


class PeakDetector:
    def __init__(self):
        pass

    def detect_peaks(self, x):
        """Returns peak indices"""
        return signal.find_peaks(x)[0]
