from abc import ABC, abstractmethod


class RiskClassificationInputMetric(ABC):
    @abstractmethod
    def generate_metric(self):
        pass
