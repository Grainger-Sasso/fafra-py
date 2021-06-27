import unittest
import random
from src.risk_classification.input_metrics.metric_instances import sma_metric

class TestSMA(unittest.TestCase):
    def setUp(self):
        self.SMA = Metric()

class TestSMAMetric(TestSMA):
    def test_generate_metric(self):
        for i in range(1, 10):
            mysum = 0
            values = []
            for i in range(1, 30):
                random_num = random.uniform(-100, 100)
                mysum += random_num
                values.append(random_num)
            self.assertEqual(self.SMA.generate_metric(values), mysum)
if __name__ == '__main__':
    unittest.main()
