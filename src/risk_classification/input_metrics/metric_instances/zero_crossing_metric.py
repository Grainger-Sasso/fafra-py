import numpy as np
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
 

METRIC_NAME = MetricNames.ZERO_CROSSING


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super(RiskClassificationInputMetric, self).__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        return self._find_zero_crossing(kwargs['data'], kwargs['sampling_frequency'],enable_mean_crossing = 0)
    
    def _find_zero_crossing(self, data, sampling_frequency,enable_mean_crossing = 0):
        """
        zero crossing
        :param data:input would be a 1D array
        :param sampling_frequency:would be a single float number
        :return:the zero crossing or mean crossing for an array
        """
        data_array = np.array(data)
        real_len = len(data_array)-1
        if real_len < 1:
            raise TooFewDataError
        if enable_mean_crossing:
            data_array = data_array-np.mean(data_array)
        result = data_array[1:len(data_array)]*data_array[0:real_len]
        for idx, elem in enumerate(result):
            if elem >= 0:
                result[idx] = 0 
            else:
                result[idx] = 1
        result = result.sum()/((1/sampling_frequency)-1)
        return result
class Error(Exception):
    """Base class for other exceptions"""
    pass
class TooFewDataError(Error):
    """Raised when input data array has too few elements"""
    pass

        
