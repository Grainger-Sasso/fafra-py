import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_data_types import MetricDataTypes
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.SIGNAL_ENERGY
METRIC_DATA_TYPE = MetricDataTypes.VERTICAL


class Metric(RiskClassificationInputMetric):
    def __init__(self):
         super().__init__(METRIC_NAME, METRIC_DATA_TYPE)

    def generate_metric(self, **kwargs):
        # convert the data into numpy array 
        data_array = np.array(kwargs['data'])             
        # take absolute value and square each element
        data_array = np.abs(data_array)
        #return the sum of the squared result to find the signal energy metric
        data_array = np.square(data_array)
        return np.sum(data_array, dtype=np.float64)



