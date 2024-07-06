import numpy as np


class AutoCorrelation:
    
    def __init__(self):
        pass

    def autocorrelate(self, x: np.array):
        # return np.correlate(x, x, mode='full')[len(x)//2:]
        ac_results = np.correlate(x, x, mode='full')
        ac_results = ac_results[ac_results.size // 2:]
        lags = np.linspace(0, len(ac_results), len(ac_results), endpoint=False, dtype=int)
        return lags, ac_results



