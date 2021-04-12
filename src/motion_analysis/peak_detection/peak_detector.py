from scipy import signal


class PeakDetector:
    def __init__(self):
        pass

    def detect_peaks(self, x):
        signal.find_peaks(x)
