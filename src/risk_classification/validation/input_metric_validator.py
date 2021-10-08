import shap

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier

class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self):
        pass

    def perform_shap_values(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        model.train_model(x_train, y_train)
        m = model.get_model()
        # explain the model's predictions using SHAP
        explainer = shap.KernelExplainer(m.predict, x_test)
        shap_values = explainer.shap_values(x_test)

        # visualize the first prediction's explaination
        shap.summary_plot(shap_values, x_test)

