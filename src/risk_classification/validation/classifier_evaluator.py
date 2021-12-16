import os
import numpy as np
import json
from typing import List
import uuid

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
        final_results = []
        # Iterate evaluation through list of classifiers
        #[{'classifier_name': 'KNN', 'eval_metrics':[{'metric_name': 'PFI', 'metric_value': {'gait_speed': 0.542,...}},
        # {'metric_name': 'SHAP', 'metric_value': 4.2112}]}, {}]
        for classifier in classifiers:
            result_dictionary = {}
            classifier_name = classifier.get_name() #getting name of classifier, add this to the dictionary
            # Create dir at output_path level
            classifier_path = os.path.join(output_path, str('classifier_' + classifier_name + '_' + uuid.uuid4()))
            os.mkdir(classifier_path)
            result_dictionary['classifier_name'] = classifier_name
            result_dictionary['eval_metrics'] = []
            metric_dictionary = {}
            plots_dict = {}
            # Perform specified evaluation techniques on classifier
            for eval_metric in eval_metrics:
                metric_name = eval_metric.get_name() #getting name of the input metrics, PFI/SHAP
                metric_dictionary['metric_name'] = metric_name
                if eval_metric == ClassifierMetrics.PFI:
                    y = input_metrics.get_labels()
                    results = self.input_metric_validator.perform_permutation_feature_importance(
                            classifier, input_metrics, y)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.SHAP:
                    x = input_metrics.get_metric_matrix()
                    results = self.input_metric_validator.perform_shap_values(
                            classifier, x)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.CV:
                    x = input_metrics.get_metric_matrix()
                    y = input_metrics.get_labels()
                    results = classifier.cross_validate(x, y)
                    metric_dictionary['metric_value'] = results
                else:
                    raise ValueError(f'Evaluation metric provided, {eval_metric}'
                                     f', is not a valid metric: {ClassifierMetrics.get_all_values()}')
                result_dictionary['eval_metrics'].append(metric_dictionary)
            final_results.append(result_dictionary)
            # Call function to write out the plots
            self.write_results_plots(classifier_path, results['plots'])
            self.write_results_json(classifier_path, results['metrics'])
        print('complete')
        return results

    def write_results_plot(self, classier_path, plots):
        #call a json library that can write a json object to a json file
        # 'C://user.../classifier_name/eval_metrics/results.json'
        plots_output_path = os.path.join(classier_path, 'plots')
        for plot in plots:
            # Building path to the file w/extension
            metric_name = plot.key()

            with open('', 'w') as plot_file:
                pass
                # Statement to write the plot out to plot_file

    def write_results_json(self, classier_path, metrics_result):
        #call a json library that can write a json object to a json file
        eval_metrics_output_path = os.path.join(classier_path, 'eval_metrics')
        os.mkdir(eval_metrics_output_path)
        full_path = os.path.join(eval_metrics_output_path, 'results.json')
        with open(full_path, 'w') as json_file:
            json.dump(metrics_result, json_file)


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

