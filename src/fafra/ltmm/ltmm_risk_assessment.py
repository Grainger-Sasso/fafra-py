import numpy as np
from typing import List
from enum import Enum
from sklearn.model_selection import train_test_split

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
        self.input_metric_names = RiskClassificationMetricNames
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
        y_prediction, y_test = self._make_model_predictions(faller_metrics, faller_status, non_faller_metrics, non_faller_status)
        # Plot the trained risk features
        # TODO: create a means to visualize the data prior to model training
        # self._viz_trained_model(model_training_x, model_training_y)
        # Make inference on the cohort
        # Output inferences to csv
        return self._compare_prediction_to_test(y_prediction, y_test)

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

    def _make_model_predictions(self, faller_metrics, faller_status, non_faller_metrics, non_faller_status):
        input_x = faller_metrics + non_faller_metrics
        input_y = faller_status + non_faller_status
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.33, random_state=42)
        # Split the data into test and train categories
        # Train risk model on features
        self._train_risk_model(x_train, y_train)
        y_prediction = self.risk_classifier.make_prediction(x_test)
        return y_prediction, y_test

    def _viz_trained_model(self, x, y):
        self.rc_viz.plot_classification(self.risk_classifier.get_model(), x, y)

    def _train_risk_model(self, x, y):
        self.risk_classifier.generate_model()
        self.risk_classifier.fit_model(x, y)

    def _initialize_dataset(self):
        self.ltmm_dataset.generate_header_and_data_file_paths()
        self.ltmm_dataset.read_dataset()


class RiskClassificationMetricNames(Enum):
    fft_peak_x_value = 'fft_peak_x_value'
    fft_peak_y_value = 'fft_peak_y_value'
    rms = 'rms'
    mean = 'mean'
    std_dev = 'std_dev'


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
