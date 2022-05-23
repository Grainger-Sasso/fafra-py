import os
from pickle import FALSE
import numpy as np
import json
from typing import List
import uuid
import matplotlib.pyplot as plt
from os.path import exists
import shap

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
        results={}
        # Iterate evaluation through list of classifiers
        #[{'classifier_name': 'KNN', 'eval_metrics':[{'metric_name': 'PFI', 'metric_value': {'gait_speed': 0.542,...}},
        # {'metric_name': 'SHAP', 'metric_value': 4.2112}]}, {}]
        for classifier in classifiers:
            result_dictionary = {}
            classifier_name = classifier.get_name() #getting name of classifier, add this to the dictionary
            # Create dir at output_path level
            classifier_path = os.path.join(output_path, str('classifier_' + classifier_name + '_' + str(uuid.uuid4())))
            if exists(classifier_path)==False:
                os.mkdir(classifier_path)
            result_dictionary['classifier_name'] = classifier_name
            result_dictionary['eval_metrics'] = []
            metric_dictionary = {}
            plots_dict = {}
            self.eval_ms=eval_metrics
            # Perform specified evaluation techniques on classifier
            for eval_metric in eval_metrics:
                metric_name = eval_metric.get_name() #getting name of the input metrics, PFI/SHAP
                self.mn=metric_name
                metric_dictionary['metric_name'] = metric_name
                if eval_metric == ClassifierMetrics.PFI:
                    y = input_metrics.get_labels()
                    results = self.input_metric_validator.perform_permutation_feature_importance(
                            classifier, input_metrics, y)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.SHAP:
                    #x = input_metrics.get_metric_matrix()
                    results = self.input_metric_validator.perform_shap_values(
                            classifier, input_metrics)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.SHAP_GBM:
                    #x = input_metrics.get_metric_matrix()
                    results = self.input_metric_validator.perform_shap_values_gbm(
                            classifier, input_metrics)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.PDP_KNN:
                    results = self.input_metric_validator.perform_partial_dependence_plot_knn(
                            classifier, input_metrics)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.PDP_GBM:
                    results = self.input_metric_validator.perform_partial_dependence_plot_lightGBM(
                            classifier, input_metrics)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.LIME:
                    results = self.input_metric_validator.perform_lime(
                            classifier, input_metrics,50)
                    metric_dictionary['metric_value'] = results['metrics']
                    plots_dict[eval_metric] = results['plots']
                elif eval_metric == ClassifierMetrics.CV:
                    x,name = input_metrics.get_metric_matrix()
                    y = input_metrics.get_labels()
                    res = classifier.cross_validate(x, y)
                    metric_dictionary['metric_value'] = res
                    for r in res:
                        if type(res[r]) ==np.ndarray:
                            res[r]=res[r].tolist()  
                    results['plots']=None
                    results['metrics']=res
                else:
                    raise ValueError(f'Evaluation metric provided, {eval_metric}'
                                     f', is not a valid metric: {ClassifierMetrics.get_all_values()}')
                # Call function to write out the plots
                self.write_results_plot(classifier_path, results['plots'],eval_metric)
                self.write_results_json(classifier_path, results['metrics'],eval_metric)
                result_dictionary['eval_metrics'].append(metric_dictionary)
            final_results.append(result_dictionary)
            #self.write_results_plot(classifier_path, plots_dict)
            #self.write_results_json(classifier_path, metric_dictionary)
        print('complete')
        return results

    def write_results_plot(self, classier_path, plots,eval_metric):
        #call a json library that can write a json object to a json file
        # 'C://user.../classifier_name/eval_metrics/results.json'
        plots_output_path = os.path.join(classier_path, str(self.mn)+'plots ')
        if plots==None or len(plots)==0:
            return
        if eval_metric== ClassifierMetrics.PDP_GBM:
            plots_output_path=os.path.join(classier_path, 'PDP_GBM')
            if exists(plots_output_path)==False:
                os.mkdir(plots_output_path)
            #plots_output_path=os.path.join(plots_output_path,str(plot))
            
        for plot in plots:
            print(eval_metric,(eval_metric== ClassifierMetrics.PDP_GBM))
            # Building path to the file w/extension
            # if eval_metric == ClassifierMetrics.LIME:
            #     with open("KNNdata2.html", "w") as file:
            #         file.write(plot)
            #     continue
            if eval_metric== ClassifierMetrics.PDP_GBM:
                plots[plot].savefig(os.path.join(plots_output_path,str(plot)))
            elif eval_metric == ClassifierMetrics.SHAP or eval_metric==ClassifierMetrics.SHAP_GBM or eval_metric==ClassifierMetrics.PDP_KNN:
                plot.savefig(plots_output_path)
            elif eval_metric == ClassifierMetrics.LIME:
                with open(plots_output_path, "w", encoding='utf-8') as file:
                    file.write(plot)
            else:
                plot[0].figure.savefig(plots_output_path)
                plt.clf()
                # Statement to write the plot out to plot_file

    def write_results_json(self, classier_path, metrics_result,eval_metric):
        #call a json library that can write a json object to a json file
        eval_metrics_output_path = os.path.join(classier_path, str(self.mn)+'eval_metrics')
        if exists(eval_metrics_output_path)==False:
            os.mkdir(eval_metrics_output_path)
        full_path = os.path.join(eval_metrics_output_path, 'results.json')
        # if eval_metric==ClassifierMetrics.SHAP:
        #     for k in metrics_result:
        #         metrics_result[k]=metrics_result[k].tolist()
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
    output_path = r'F:\long-term-movement-monitoring-database-1.0.0\output_dir'
    cl_ev.run_models_evaluation(eval_metrics, classifiers, input_metrics, output_path)


if __name__ == '__main__':
    main()

