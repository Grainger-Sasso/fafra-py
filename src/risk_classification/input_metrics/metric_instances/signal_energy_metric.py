import numpy as np

from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


METRIC_NAME = MetricNames.SIGNAL_ENERGY


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super(RiskClassificationInputMetric, self).__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        data_array = np.array(kwargs['data'])             # convert the data into numpy array
                                                          #take absolute value and square each element
        data_array = np.abs(data_array)
        data_array = np.square(data_array)
        return np.sum(data_array)                         #return the sum of the squared result to find the signal energy metric



