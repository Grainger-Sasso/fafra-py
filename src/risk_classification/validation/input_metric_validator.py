import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier


class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self, model: Classifier,
                                               input_metrics: InputMetrics, y):
        x_test, names = input_metrics.get_metric_matrix()
        r = permutation_importance(model, x_test, y, scoring='accuracy',n_repeats=100)
        importance = r.importances_mean
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
