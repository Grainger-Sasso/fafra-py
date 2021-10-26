import shap
import lime

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier

class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self):
        pass

    def perform_lime(self, model, input_metrics: InputMetrics, value):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        cv, name = input_metrics.get_metric_matrix()

        model.train_model(x_train, y_train)
        m = model.get_model()

        names = []  # without this, won't get feature names
        for i in name:
            names.append(i.get_value())

        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=names, discretize_continuous=True)

        exp = explainer.explain_instance(x_test[value], m.predict_proba, top_labels=1)
        #exp.show_in_notebook(show_table=True, show_all=False).display()

        a = exp.as_html(show_table=True, show_all=False)
        with open("KNNdata2.html", "w") as file:
            file.write(a)

    def perform_shap_values(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        model.train_model(x_train, y_train)
        m = model.get_model()
        # explain the model's predictions using SHAP
        explainer = shap.KernelExplainer(m.predict, x_test)
        shap_values = explainer.shap_values(x_test)

        # visualize the first prediction's explaination
        cv, name = input_metrics.get_metric_matrix()
        shap.summary_plot(shap_values, x_test, feature_names=name)

