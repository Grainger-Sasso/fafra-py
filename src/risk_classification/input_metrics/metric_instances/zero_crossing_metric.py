import numpy as np
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
 

METRIC_NAME = MetricNames.ZERO_CROSSING
METRIC_DATA_TYPE = MetricDataTypes.VERTICAL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        return self._find_zero_crossing(kwargs['data'])
    
    def _find_zero_crossing(self, data, enable_mean_crossing=True):
        """
        zero crossing
        :param data:input would be a 1D array that indicates the data array
        :return:the zero crossing or mean crossing for an array
        """
        data_array = np.array(data)
        real_len = len(data_array)-1
        # perform error checking if input data too short
        if real_len < 1:
            raise ValueError('Too few data for zero-crossing metric')
        sampling_frequency=1/len(data_array)
        data_array = data_array-np.mean(data_array)
        # calculte if there is zero crossing by multiply nearby elements
        result = data_array[1:len(data_array)]*data_array[0:real_len]
        # count zero crossing frequency
        for idx, elem in enumerate(result):
            if elem >= 0:
                result[idx] = 0 
            else:
                result[idx] = 1
        # find final crossing rate by dividing time internal
        result = result.sum()/((1/sampling_frequency)-1)
        return result
