from enum import Enum
 

class MetricNames(Enum):
    AUTOCORRELATION = 'ac'
    FAST_FOURIER_TRANSFORM = 'fft'
    MEAN = 'mean'
    ROOT_MEAN_SQUARE = 'rms'
    STANDARD_DEVIATION = 'std'
    SIGNAL_ENERGY = 'se'
    COEFFICIENT_OF_VARIANCE = 'cov'
    ZERO_CROSSING = 'zc'
    SIGNAL_MAGNITUDE_AREA = 'sma'
    GAIT_SPEED_ESTIMATOR = 'gse'

    def get_name(self):
        # Self is the member here
        return self.name

    def get_value(self):
        # Self is the member here
        return self.value

    @classmethod
    def get_all_enum_entries(cls):
        return [metric_name for metric_name in cls]

    @classmethod
    def get_all_names(cls):
        return [metric_name.name for metric_name in cls]

    @classmethod
    def get_all_values(cls):
        return [metric_name.value for metric_name in cls]
