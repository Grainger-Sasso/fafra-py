import numpy as np
from typing import List
from enum import Enum


from src.datasets.ltmm.ltmm_dataset import LTMMDataset
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.input_metrics.risk_classification_input_metric import RiskClassificationInputMetric
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.fafra.ltmm.metric_generator import MetricGenerator


#TODO make a parent class for risk assessments to be used in main FAFRA so we can use generic calls to methods like initialize_dataset, assess_cohort_risk, etc
class LTMMRiskAssessment:
    def __init__(self, ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path):
        self.ltmm_dataset = LTMMDataset(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
        self._initialize_dataset()
        self.motion_filters = MotionFilters()
        self.risk_classifier = SVMRiskClassifier()
        self.rc_viz = ClassificationVisualizer()
        self.input_metrics: List[RiskClassificationInputMetric] = []
        self.metric_generator = MetricGenerator()

    def assess_cohort_risk(self):
        # TODO: JFC please refactor this to consolidate it and make it readable by humans
        # Filter the data
        self.apply_lp_filter()
        # Segment the datasets into smaller epochs to have a greater number of data points
        self.ltmm_dataset.segment_dataset(10.0)
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Perform feature extraction
        faller_metrics, faller_status = self.metric_generator.generate_metrics(ltmm_faller_data)
        non_faller_metrics, non_faller_status = self.metric_generator.generate_metrics(ltmm_non_faller_data)
        # Split input metrics into train and test
        x_train, x_test, y_train, y_test = self._generate_test_train_groups(faller_metrics, faller_status,
                                                                            non_faller_metrics, non_faller_status)
        # Fit the metric scaler to the training data
        self.risk_classifier.fit_scaler(x_train)
        # Transform the train and test input metrics
        x_train_transformed = self.risk_classifier.transform_data(x_train)
        x_test_transformed = self.risk_classifier.transform_data(x_test)
        # Evaluate model's predictive capability with k-fold cross-validation
        cv_results = self.risk_classifier.cross_val_model(x_train_transformed, y_train)
        print(cv_results)
        # Perform cross-validation for model
        # y_prediction = self._make_model_predictions(x_test)
        # print(self._score_model_performance(x_test, y_test))
        # TODO: Output inferences to csv
        # return self._compare_prediction_to_test(y_prediction, y_test)

    def apply_lp_filter(self):
        for ltmm_data in self.ltmm_dataset.get_dataset():
            lpf_data_all_axis = []
            sampling_rate = ltmm_data.get_sampling_frequency()
            data = ltmm_data.get_data()
            for axis in data.T:
                lpf_data_all_axis.append(self.motion_filters.apply_lpass_filter(axis, sampling_rate))
            lpf_data_all_axis = np.array(lpf_data_all_axis).T
            ltmm_data.set_data(lpf_data_all_axis)

    def _compare_prediction_to_test(self, y_predition, y_test):
        results = []
        for yp, yt in zip(y_predition, y_test):
            if yp == yt:
                results.append(1)
            else:
                results.append(0)
        perf = sum(results)/len(results)
        return perf

    def _generate_test_train_groups(self, faller_metrics, faller_status, non_faller_metrics, non_faller_status):
        input_x = faller_metrics + non_faller_metrics
        input_y = faller_status + non_faller_status
        x_train, x_test, y_train, y_test = self.risk_classifier.split_input_metrics(input_x, input_y)
        return x_train, x_test, y_train, y_test

    def _score_model_performance(self, x_test, y_test):
        return self.risk_classifier.score_model(x_test, y_test)

    def _viz_trained_model(self, x):
        self.rc_viz.plot_classification(self.risk_classifier.get_model(), x)


    def _initialize_dataset(self):
        self.ltmm_dataset.generate_header_and_data_file_paths()
        self.ltmm_dataset.read_dataset()


def main():
    # ltmm_dataset_name = 'LTMM'
    ltmm_dataset_name = 'LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)
    # data_before_lpf = ltmm_ra.ltmm_dataset.get_dataset()[1].get_data().T[0,:]
    perf = ltmm_ra.assess_cohort_risk()
    print(perf)
    # data_after_lpf = ltmm_ra.ltmm_dataset.get_dataset()[1].get_data().T[0,:]
    # plot_before = PlottingData(data_before_lpf, 'Unfiltered', '')
    # plot_after = PlottingData(data_after_lpf, 'Filtered', '')
    # viz = MotionVisualizer()
    # viz.plot_data([plot_before, plot_after])


if __name__ == '__main__':
    main()
