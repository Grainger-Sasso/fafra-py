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
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


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

    def test_model(self, metric_path, clin_demo_path,
                   cv, multiclass, n_splits, output_path):
        input_metrics = self.import_metrics(metric_path)
        clin_demo_data = self.read_clin_demo_file(clin_demo_path)
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        if not multiclass:
            y = self.cast_labels_bin(y)
        groups = np.array(input_metrics.get_user_ids())
        mono_groups = self.map_groups(groups)
        # Characterize dataset
        self.characterize_dataset(x, y, groups, clin_demo_data, output_path)
        # TODO: PICKUP: Plot metric distribution for users and classes
        # Evaluate classification performance
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

    def characterize_dataset(self, x, y, groups, clin_demo_data, output_path):
        ds_elements = {}
        group_demo_data = self.get_group_clin_data(clin_demo_data, groups)
        # Total number of samples
        ds_elements['total_n_samples'] = len(x)
        # Samples by fall category
        ds_elements['cat_nonfaller_n_samples'] = len([i for i in group_demo_data if int(i.get_faller_status()) == 0])
        ds_elements['cat_single_faller_n_samples'] = len([i for i in group_demo_data if int(i.get_faller_status()) == 1])
        ds_elements['cat_recurrent_faller_n_samples'] = len([i for i in group_demo_data if int(i.get_faller_status()) == 2])
        ds_elements['cat_all_faller_n_samples'] = ds_elements['cat_single_faller_n_samples'] + ds_elements['cat_recurrent_faller_n_samples']
        # Samples by gender
        ds_elements['gender_male_n_samples'] = len([i for i in group_demo_data if int(i.get_sex()) == 0])
        ds_elements['gender_female_n_samples'] = len([i for i in group_demo_data if int(i.get_sex()) == 1])
        # Samples by health status
        ds_elements['status_healthy_young_n_samples'] = len(
            [i for i in group_demo_data if 100 <= int(i.get_id()) < 200])
        ds_elements['status_healthy_older_n_samples'] = len(
            [i for i in group_demo_data if 200 <= int(i.get_id()) < 300])
        ds_elements['status_older_hypertensive_n_samples'] = len(
            [i for i in group_demo_data if 300 <= int(i.get_id())])
        # Export JSON data to results folder
        filename = 'dataset_characteristics_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        file_path = os.path.join(output_path, 'dataset_characterization', filename)
        with open(file_path, 'w') as f:
            json.dump(ds_elements, f)
        return file_path

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

    def plot_confusion_matrix(self, conf_matrix, y_test):
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

    def read_clin_demo_file(self, clinical_demo_path):
        # with open(clinical_demo_path, 'r') as file:
        #     data = []
        #     reader = csv.reader(file, dialect='excel')
        #     for row in reader:
        #         data.append(row)
        #     file.close()
        xl_file = pd.ExcelFile(clinical_demo_path)

        clinical_demo_data = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}['Baseline data']
        return clinical_demo_data

    def get_group_clin_data(self, clinical_demo_data, subj_ids):
        """
        Method to index the dataset clinical demographic data for subject clinical demographic data. Below are the
        columns of the Data_CHI2021_Carapace.xlsx demographic data file:

        'ID', 'Age (yrs)', 'Gender (0=M, 1=F)', 'Height(m)', 'Weight(kg)',
       'Cohort (HYA = young, HOA =older, HTN = older with hypertension)',
       'Comfortable Walking Speed (m/s)', 'Treadmill Belt Speed (m/s)',
       'MoCA (composite score for Montreal Cognitive Assessment)',
       'TMT_A_time (s)', 'TMT_B_time (s)', 'Mini-BEST (composite score)',
       'Repeated  Sit To Stand time (s)',
       'Faller_Cohort ( 0 = No Fall, 1 = Fall, 2 = Recurrent faller with 2 or more)'

        :param clinical_demo_data: dataset clinical demographic data
        :param subj_id: ID of the subject
        :param trial_id: ID of the trial
        :return: subject clinical demographic data
        """
        group_clin_data = []
        for id in subj_ids:
            subj_ix = clinical_demo_data[clinical_demo_data['ID'] == int(id)].index[0]
            cols = clinical_demo_data.columns
            subj_data = {}
            for col in cols:
                subj_data[col] = clinical_demo_data.loc[subj_ix, col]
            id = subj_data['ID']
            age = subj_data['Age (yrs)']
            sex = subj_data['Gender (0=M, 1=F)']
            faller_status = subj_data['Faller_Cohort ( 0 = No Fall, 1 = Fall, 2 = Recurrent faller with 2 or more)']
            height = subj_data['Height(m)']
            trial = ''
            weight = subj_data['Weight(kg)']
            group_clin_data.append(ClinicalDemographicData(id, age, sex, faller_status, height, trial,
                                       name='', weight=weight, other=subj_data))
        return group_clin_data

    def export_classifier(self, model_output_path, model_name, scaler_name):
        model_name = model_name + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
        scaler_name = scaler_name + time.strftime("%Y%m%d-%H%M%S") + '.bin'
        model_path = os.path.join(model_output_path, model_name)
        scaler_path = os.path.join(model_output_path, scaler_name)
        # self.rc.model.save_model(model_path)
        joblib.dump(self.rc.get_model(), model_path)
        joblib.dump(self.rc.get_scaler(), scaler_path)
        return model_path, scaler_path


def main():
    mt = ModelTrainer()
    # LM path
    # path = r'F:\long-term-movement-monitoring-database-1.0.0\input_metrics\model_input_metrics_20230116-135200.json'#'/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/features/model_input_metrics_20230116-135200.json'
    # GS path
    model_path = r'/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/features/model_input_metrics_20230116-135200.json'
    clin_demo_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/participant_metadata/Data_CHI2021_Carapace.xlsx'
    output_path = '/home/grainger/Desktop/skdh_testing/uiuc_ml_analysis/results/'
    multiclass = True
    n_splits = 5
    # cv = KFold(n_splits=n_splits, shuffle=True)
    # cv = GroupShuffleSplit(n_splits=n_splits)
    # cv = GroupKFold(n_splits=n_splits)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    mt.test_model(model_path, clin_demo_path, cv, multiclass, n_splits, output_path)
    # mt.assess_input_feature(path,r'F:\long-term-movement-monitoring-database-1.0.0\output_dir')


#D:\carapace\fafra-py\validation\ml_evaluation\uiuc_walking_dataset\uiuc_model_trainer.py
if __name__ == '__main__':
    main()
