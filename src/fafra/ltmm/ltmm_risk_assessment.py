import numpy as np
import time
from typing import List, Tuple


from src.datasets.ltmm.ltmm_dataset import LTMMDataset
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.validation.cross_validator import CrossValidator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.fafra.ltmm.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.metric_names import MetricNames


#TODO make a parent class for risk assessments to be used in main FAFRA so we can use generic calls to methods like initialize_dataset, assess_cohort_risk, etc
class LTMMRiskAssessment:
    def __init__(self, ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path,
                 report_home_75h_path, input_metric_names: Tuple[MetricNames]):
        self.ltmm_dataset = LTMMDataset(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
        self.filt = MotionFilters()
        self.rc = SVMRiskClassifier()
        self.rc_viz = ClassificationVisualizer()
        self.metric_names: Tuple[MetricNames] = input_metric_names
        self.mg = MetricGenerator()
        self.cv = CrossValidator()

    def assess_model_accuracy(self):
        # Preprocess the data
        self.preprocess_data()
        # Generate risk metrics
        x, y = self.generate_risk_metrics()
        # Split input data into test and train groups
        x_train, x_test, y_train, y_test = self._generate_test_train_groups(x, y)
        # Fit model to training data
        self.rc.fit_model(x_train, y_train)
        # Make predictions on the test data
        y_predictions = self.rc.make_prediction(x_test)
        # Compare predictions to test
        return self.rc.create_classification_report(y_test, y_predictions)

    def cross_validate_model(self, k_folds=5):
        # Preprocess the data
        self.preprocess_data()
        # Generate risk metrics
        x, y = self.generate_risk_metrics()
        # Evaluate model's predictive capability with k-fold cross-validation
        return self.cv.cross_val_model(self.rc.get_model(), x, y, k_folds)

    def preprocess_data(self):
        # Filter the data
        self.apply_lp_filter()
        # self.apply_kalman_filter()
        # Remove effects of gravity in vertical axis
        self.norm_vert_axis()
        # Segment the datasets into smaller epochs to have a greater number of data points
        self.ltmm_dataset.segment_dataset(10.0)

    def norm_vert_axis(self):
        for ltmm_data in self.ltmm_dataset.get_dataset():
            tri_ax_data = ltmm_data.get_data()
            vert_ax_data = np.array(tri_ax_data.T[0])
            norm_vert_ax_data = np.array([x - 1.0 for x in vert_ax_data])
            tri_ax_data[:, 0] = norm_vert_ax_data

    def generate_risk_metrics(self, scale_data=True):
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Generate metrics
        faller_metrics, faller_status = self.mg.generate_metrics(ltmm_faller_data, self.metric_names)
        non_faller_metrics, non_faller_status = self.mg.generate_metrics(ltmm_non_faller_data, self.metric_names)
        # Transform the train and test input metrics
        x = faller_metrics + non_faller_metrics
        y = faller_status + non_faller_status
        if scale_data:
            x = self.rc.scale_input_data(x)
        return x, y

    def _call_metric_generator(self, ltmm_dataset):
        faller_status = []
        dataset_metrics = []
        for ltmm_data in ltmm_dataset:
            faller_status.append(int(ltmm_data.get_faller_status()))
            dataset_metrics.append(self._derive_metrics(ltmm_data, input_metric_names))
        return list(dataset_metrics), list(faller_status)

    def apply_kalman_filter(self):
        for ltmm_data in self.ltmm_dataset.get_dataset()[:2]:
            v_y_ax_data = ltmm_data.get_axis_acc_data('vertical')
            ml_x_ax_data = ltmm_data.get_axis_acc_data('mediolateral')
            ap_z_ax_data = ltmm_data.get_axis_acc_data('anteroposterior')
            sampling_rate = ltmm_data.get_sampling_frequency()
            kf_x_ml, kf_y_v, kf_z_ap, kf_ub_y_v = self.filt.apply_kalman_filter(ml_x_ax_data, v_y_ax_data,
                                                                                ap_z_ax_data, sampling_rate)
            ltmm_data.get_data()[:, 0] = kf_ub_y_v
            ltmm_data.get_data()[:, 1] = kf_x_ml
            ltmm_data.get_data()[:, 2] = kf_z_ap
            print('kf')

    def apply_lp_filter(self):
        for ltmm_data in self.ltmm_dataset.get_dataset():
            lpf_data_all_axis = []
            sampling_rate = ltmm_data.get_sampling_frequency()
            data = ltmm_data.get_data()
            for axis in data.T:
                lpf_data_all_axis.append(self.filt.apply_lpass_filter(axis, sampling_rate))
            lpf_data_all_axis = np.array(lpf_data_all_axis).T
            ltmm_data.set_data(lpf_data_all_axis)
            print('lpf')

    def _compare_prediction_to_test(self, y_predition, y_test):
        comparison = np.array(np.array(y_predition) == np.array(y_predition),
                              dtype=int)
        return sum(comparison)/len(comparison)

    def _generate_test_train_groups(self, x, y):
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(x, y)
        return x_train, x_test, y_train, y_test


def main():
    st = time.time()
    # ltmm_dataset_name = 'LTMM'
    ltmm_dataset_name = 'LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    input_metric_names = tuple([MetricNames.GAIT_SPEED_ESTIMATOR])
    ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path,
                                 report_home_75h_path, input_metric_names)
    print(ltmm_ra.assess_model_accuracy())


    # cv_results = ltmm_ra.cross_validate_model()
    # print(cv_results)

    ft = time.time() - st
    print('##############################################################')
    print(ft)
    print('##############################################################')



if __name__ == '__main__':
    main()
