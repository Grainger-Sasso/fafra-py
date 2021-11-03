from typing import List

from src.risk_classification.validation.classifier_metrics import ClassifierMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier
from src.risk_classification.input_metrics.input_metrics import InputMetrics


class ClassifierEvaluator:
    def __init__(self):
        pass

    def run_models_evaluation(self, eval_metrics: List[ClassifierMetrics],
                              classifiers: List[Classifier],
                              input_data: InputMetrics, output_path):
        pass

