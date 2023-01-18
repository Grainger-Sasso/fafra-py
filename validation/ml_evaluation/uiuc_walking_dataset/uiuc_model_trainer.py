import time
import os
import json
import joblib
import numpy as np

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier


class ModelTrainer:
    def __init__(self):
        self.rc = LightGBMRiskClassifier({})

    def train_classifier_model(self, metric_path, model_output_path, model_name, scaler_name):
        input_metrics = self.import_metrics(metric_path)
        x, names = input_metrics.get_metric_matrix()
        names = [name.replace(':', '_') for name in names]
        names = [name.replace(' ', '_') for name in names]
        names = [name.replace('__', '_') for name in names]
        # names = [name.replace for name in names]
        y = input_metrics.get_labels()
        # Train scaler on training data
        self.rc.scaler.fit(x)
        # Transform traning data
        x_train_t = self.rc.scaler.transform(x)
        # Train model on training data
        self.rc.train_model_optuna(x_train_t, y, names=names)
        # Export model, scaler
        # model_path, scaler_path = self.export_classifier(model_output_path, model_name, scaler_name)
        # return model_path, scaler_path
        return self.rc

    def export_classifier(self, model_output_path, model_name, scaler_name):
        model_name = model_name + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
        scaler_name = scaler_name + time.strftime("%Y%m%d-%H%M%S") + '.bin'
        model_path = os.path.join(model_output_path, model_name)
        scaler_path = os.path.join(model_output_path, scaler_name)
        # self.rc.model.save_model(model_path)
        joblib.dump(self.rc.get_model(), model_path)
        joblib.dump(self.rc.get_scaler(), scaler_path)
        return model_path, scaler_path

    def test_model(self, metric_path):
        input_metrics = self.import_metrics(metric_path)
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(input_metrics)
        x_train, x_test = self.rc.scale_train_test_data(x_train, x_test)
        num_classes = 3
        self.rc.train_model_optuna_multiclass(x_train, y_train, num_classes, names=names)


        # Make predictions and generate confusion matrix
        # y_pred = self.rc.make_prediction(x_test, True)
        # cm = self.rc.multilabel_confusion_matrix(y_test, y_pred)

        # Score the model
        acc, pred = self.rc.score_model(x_test, y_test, True)
        cr = self.rc.create_classification_report(y_test, pred)

        print('ok')


    def assess_input_features(self, metrics_path, output_path):
        input_metrics = self.import_metrics(metrics_path)
        # Calls to lime, pdp ,shap, etc

    def import_metrics(self, path) -> InputMetrics:
        with open(path, 'r') as f:
            input_metrics = json.load(f)
        im = self.finalize_metric_formatting(input_metrics)
        return im

    def finalize_metric_formatting(self, metric_data):
        im = InputMetrics()
        for name, metric in metric_data['metrics'][0].items():
            name = self.format_name(name)
            metric = InputMetric(name, np.array(metric))
            im.set_metric(name, metric)
        im.labels = np.array(metric_data['labels'])
        im.user_ids = metric_data['user_ids']
        im.trial_ids = metric_data['trial_ids']
        return im

    def format_name(self, name):
        name = name.replace(':', '_')
        name = name.replace(' ', '_')
        name = name.replace('__', '_')
        return name

def main():
    mt = ModelTrainer()
    path = '/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/features/model_input_metrics_20230116-135200.json'
    mt.test_model(path)


if __name__ == '__main__':
    main()
