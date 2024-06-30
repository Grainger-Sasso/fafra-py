import unittest
from src.risk_classification.input_metrics.metric_instances import cov_metric
from src.risk_classification.input_metrics.metric_instances import std_metric
from src.risk_classification.input_metrics.metric_instances import mean_metric
import random

class Test_COV(unittest.TestCase):
    def setUp(self):
        self.COV = cov_metric.Metric()

class Test_COV_Metric(Test_COV):
    def test_generate_metric(self):
        std_test, mean_test = std_metric.Metric(), mean_metric.Metric()
        for i in range(1, 10):
            kwargs = {'data' : [random.uniform(-100, 100) for j in range(1, 30)]}
            self.assertEqual(self.COV.generate_metric(**kwargs),
                             std_test.generate_metric(**kwargs) / mean_test.generate_metric(**kwargs))

if __name__ == '__main__':
    unittest.main()
