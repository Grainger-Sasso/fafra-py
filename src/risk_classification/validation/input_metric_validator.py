import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier


class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self, model: Classifier,
                                               input_metrics: InputMetrics, y,
                                               show_plot=False):
        x_test, names = input_metrics.get_metric_matrix()
        r = permutation_importance(model.get_model(), x_test, y, scoring='accuracy',n_repeats=50)
        importance = r.importances_mean
        y_pos = np.arange(len(importance))
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        bar_plot = plt.bar([x for x in range(len(importance))], importance)
        bar_plot.xticks(y_pos, names, color='orange', rotation=15, fontweight='bold', fontsize='5', horizontalalignment='right')
        bar_plot.xlabel('feature Metrics', fontweight='bold', color='blue', fontsize='5', horizontalalignment='center')
        if show_plot:
            plt.show()
        return [bar_plot, r, importance]

    def perform_shap_values(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        model.train_model(x_train, y_train)
        m = model.get_model()
        # explain the model's predictions using SHAP
        explainer = shap.KernelExplainer(m.predict, x_test)
        shap_values = explainer.shap_values(x_test)

        # visualize the first prediction's explaination
        cv,name = input_metrics.get_metric_matrix()
        shap_plot = shap.summary_plot(shap_values, x_test, feature_names=name)

        #p=shap.force_plot(explainer.expected_value, shap_values[0:5,:],x_test[0:5,:])
        # p = shap.force_plot(explainer.expected_value, shap_values,x_test, matplotlib = True, show = False)
        # plt.savefig('tmp.svg')
        # plt.close()
        #shap.plots.force(shap_values)

        return [shap_plot, shap_values]
