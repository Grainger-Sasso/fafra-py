import shap

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier

class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self):
        pass

    def perform_shap_values(model: Classifier, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        trained_model = model.fit(x_train, y_train)

        # explain the model's predictions using SHAP
        explainer = shap.Explainer(trained_model)
        shap_values = explainer(x_train)

        # visualize the first prediction's explaination
        shap.plots.waterfall(shap_values[0])

