import unittest

from src.risk_classification.input_metrics.metric_instances.signal_energy_metric import Metric
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric


class TestSingalEnergyMetric(unittest.TestCase):
    def test_singal_energy_metric(self):
        test_data = [1, 2, 3, 4]
        result = Metric().generate_metric(data=test_data)
        self.assertEqual(result, 30)

if __name__ == '__main__':
    unittest.main()
