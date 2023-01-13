import time
import os
import joblib

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier


class ModelTrainer:
    def __init__(self):
        self.rc = LightGBMRiskClassifier({})

    def train_classifier_model(self, input_metrics: InputMetrics, model_output_path, model_name, scaler_name):
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

    def test_model(self, input_metrics):
        x, names = input_metrics.get_metric_matrix()
        names = [name.replace(':', '_') for name in names]
        names = [name.replace(' ', '_') for name in names]
        names = [name.replace('__', '_') for name in names]
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(input_metrics)
        x_train, x_test = self.rc.scale_train_test_data(x_train, x_test)
        self.rc.train_model_optuna(x_train, y_train, names=names)
        acc, pred = self.rc.score_model(x_test, y_test)
        cr = self.rc.create_classification_report(y_test, pred)
        print('ok')

