import numpy as np
from typing import List

from src.risk_classification.validation.classifier_metrics import ClassifierMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.validation.input_metric_validator import InputMetricValidator
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier


class ClassifierEvaluator:
    def __init__(self):
        self.input_metric_validator = InputMetricValidator()

    def run_models_evaluation(self, eval_metrics: List[ClassifierMetrics],
                              classifiers: List[Classifier],
                              input_metrics: InputMetrics, output_path):
        plot_results = []
        result_metrics = []
        # Iterate evaluation through list of classifiers
        for classifier in classifiers:
            # Perform specified evaluation techniques on classifier
            for eval_metric in eval_metrics:
                if eval_metric == ClassifierMetrics.PFI:
                    y = input_metrics.get_labels()
                    results = self.input_metric_validator.perform_permutation_feature_importance(
                            classifier, input_metrics, y)
                    plot_results.append(results[0])
                    result_metrics.append(results[1:])
                elif eval_metric == ClassifierMetrics.SHAP:
                    x = input_metrics.get_metric_matrix()
                    results = self.input_metric_validator.perform_shap_values(
                            classifier, x)
                    plot_results.append(results[0])
                    result_metrics.append(results[1])
                elif eval_metric == ClassifierMetrics.CV:
                    x = input_metrics.get_metric_matrix()
                    y = input_metrics.get_labels()
                    results = classifier.cross_validate(x, y)
                    result_metrics.append(results)
                else:
                    raise ValueError(f'Evaluation metric provided, {eval_metric}'
                                     f', is not a valid metric: {ClassifierMetrics.get_all_values()}')
        self.parse_results(plot_results, result_metrics)
        self.write_results(output_path, plot_results, result_metrics)
        print('complete')
        return results

    def parse_results(self, result_plot, metrics_result):
        parsed_results = None
        return parsed_results

    def write_results(self, output_path, result_plot, metrics_result):
        pass


def main():
    cl_ev = ClassifierEvaluator()
    eval_metrics = [ClassifierMetrics.PFI]
    # classifiers = [KNNRiskClassifier(), LightGBMRiskClassifier({}),
    #                SVMRiskClassifier()]
    classifiers = [KNNRiskClassifier()]
    input_metrics = InputMetrics()
    metric_name = MetricNames.EXAMPLE
    input_metric = InputMetric(metric_name, np.array([]))
    input_metrics.set_metric(MetricNames.EXAMPLE, input_metric)
    y = np.array([])
    input_metrics.set_labels(y)
    output_path = r'C://example//path'
    cl_ev.run_models_evaluation(eval_metrics, classifiers, input_metrics, output_path)


if __name__ == '__main__':
    main()

