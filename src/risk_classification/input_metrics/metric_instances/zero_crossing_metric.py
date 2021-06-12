import numpy as np
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
 

METRIC_NAME = MetricNames.ZERO_CROSSING


class Metric(RiskClassificationInputMetric):
    def __init__(self):
        super(RiskClassificationInputMetric, self).__init__(METRIC_NAME)

    def generate_metric(self, **kwargs):
        return self._find_zero_crossing(kwargs['data'], kwargs['sampling_frequency'])
    def _find_zero_crossing(self, data, sampling_frequency):
        """
        autocorrelation peak
        :param data:
        :param sampling_frequency:
        :return:
        """
        data_array=np.array(data)
        real_len=len(data_array)-1
        if real_len<1:
            return data_array[0]
        result=data_array[1:len(data_array)]*data_array[0:real_len]
        for idx in range(0,len(result)):
            if result[idx]>=0:
                result[idx]=0
            else:
                result[idx]=1
        result=result.sum()/((1/sampling_frequency)-1)
        return result


        
