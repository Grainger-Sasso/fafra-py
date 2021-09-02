import unittest

from src.risk_classification.input_metrics.metric_instances.zero_crossing_metric import Metric
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


class TestZeroCrossing(unittest.TestCase):
    def test_zero_crossing_metric(self):
        test_data = [1, -2, 3, 4,-5]
        frequency=1/3
        result = Metric().generate_metric(data=test_data,enable_mean_crossing=1)
        self.assertEqual(result, 0.75)

if __name__ == '__main__':
    unittest.main()
