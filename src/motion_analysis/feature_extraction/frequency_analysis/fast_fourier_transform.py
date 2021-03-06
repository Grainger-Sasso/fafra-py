import numpy as np
from scipy.fft import fft


class FastFourierTransform:
    def __init__(self):
        pass

    def perform_fft(self, data: np.array):
        return fft(data)
