import time
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
import seaborn as sns
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.validation.classifier_metrics import ClassifierMetrics
from src.risk_classification.validation.classifier_evaluator import ClassifierEvaluator
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
        multiclass = True
        n_splits = 5

        # cv = KFold(n_splits=n_splits, shuffle=True)
        # cv = GroupShuffleSplit(n_splits=n_splits)
        cv = GroupKFold(n_splits=n_splits)
        # cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)

        input_metrics = self.import_metrics(metric_path)
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        if not multiclass:
            y = self.cast_labels_bin(y)
        groups = np.array(input_metrics.get_user_ids())
        mono_groups = self.map_groups(groups)
        scores, pm_mean = self.rc.group_cv(
            x, y, mono_groups, names, multiclass, cv, n_splits)

        # Make predictions and generate confusion matrix
        # x_train, x_test, y_train, y_test = self.rc.split_input_metrics(input_metrics)
        # x_train, x_test = self.rc.scale_train_test_data(x_train, x_test)
        # num_classes = 3
        # self.rc.train_model_optuna_multiclass(x_train, y_train, num_classes, names=names)

        # y_pred = self.rc.make_prediction(x_test, True)
        # cm = self.rc.multilabel_confusion_matrix(y_test, y_pred)

        # Score the model
        # acc, pred = self.rc.score_model(x_test, y_test, True)
        # cr = self.rc.create_classification_report(y_test, pred)
        for score in scores:
            print(score['performance_metrics'])
        print('\n\n')

        a = 0
        for score in scores:
            a += score['accuracy']
        print('mean accuracy: ' + str(a/5))
        print('\n\n')

        print(pm_mean)
        print('done')

    def cast_labels_bin(self, y):
        new_labels = []
        for val in y:
            if val != 0:
                new_labels.append(1)
            else:
                new_labels.append(0)
        return np.array(new_labels)

    def assess_input_feature(self, metrics_path, output_path):
        cl_ev = ClassifierEvaluator()
        eval_metrics = [ClassifierMetrics.SHAP_GBM]
        classifiers = [self.rc]
        input_metrics = self.import_metrics(metrics_path)
        x, names = input_metrics.get_metric_matrix()
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(input_metrics)
        x_train, x_test = self.rc.scale_train_test_data(x_train, x_test)
        num_classes = 3
        classifiers[0].train_model_optuna_multiclass(x_train, y_train, num_classes, names=names)
        cl_ev.run_models_evaluation(eval_metrics, classifiers, input_metrics, output_path)#the SHAP function
        y_pred = self.rc.make_prediction(x_test, True)
        cm = self.rc.multilabel_confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, y_test)
        print(self.rc.assess_roc_auc(y_test,self.rc.model.predict(x_test),1))# sample assess_roc_auc call

    
    def plot_confusion_matrix(self,conf_matrix,y_test):
        fig, ax = plt.subplots(1,3)
        class_names=list(set(y_test))
        for axes,cfs_matrix, label in zip(ax.flatten(),conf_matrix,class_names):
            disp = ConfusionMatrixDisplay(cfs_matrix, display_labels=list(set(y_test)))
            disp.plot(include_values=True, cmap="viridis", ax=axes, xticks_rotation="vertical")
            disp.im_.colorbar.remove()
        fig.colorbar(disp.im_, ax=ax)
        #plt.savefig('confusion_matrix.png')
        plt.show()

    def map_groups(self, groups):
        ix = 1
        g_i = groups[0]
        new_groups = []
        for group in groups:
            if group == g_i:
                new_groups.append(ix)
            else:
                ix += 1
                g_i = group
                new_groups.append(ix)
        return np.array(new_groups)


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
    # LM path
    # path = r'F:\long-term-movement-monitoring-database-1.0.0\input_metrics\model_input_metrics_20230116-135200.json'#'/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/features/model_input_metrics_20230116-135200.json'
    # GS path
    path = r'/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/features/model_input_metrics_20230116-135200.json'
    mt.test_model(path)
    # mt.assess_input_feature(path,r'F:\long-term-movement-monitoring-database-1.0.0\output_dir')


#D:\carapace\fafra-py\validation\ml_evaluation\uiuc_walking_dataset\uiuc_model_trainer.py
if __name__ == '__main__':
    main()
