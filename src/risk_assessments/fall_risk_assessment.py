import numpy as np
import time
import os
import glob
import importlib
import random
from typing import Tuple, Dict, List, Any
from definitions import ROOT_DIR
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.validation.cross_validator import CrossValidator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.risk_classification.risk_classifiers.classifier import Classifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier



class FallRiskAssessment:
    def __init__(self, risk_classifier):
        # Required input parameters
        self.dataset_builders: Dict[str: DatasetBuilder] = {}
        self.datasets: Dict[DatasetNames: Dataset] = {}
        self.rc: Classifier = risk_classifier
        self.filter = MotionFilters()
        self.rc_viz = ClassificationVisualizer()
        self.mg = MetricGenerator()
        self.cv = CrossValidator()
        self.scaler: StandardScaler = StandardScaler()

    def perform_risk_assessment(self, dataset_info: List[Dict[str, Any]],
                                input_metric_names: Tuple[MetricNames]):
        # input_metrics
        # Build datasets from given names
        self._build_datasets(dataset_info)
        # Preprocess imu data
        self._preprocess_data()
        # Derive risk metrics
        random.shuffle(self.datasets[DatasetNames.LTMM].get_dataset())
        x, y = self.generate_risk_metrics(input_metric_names)
        path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\test_data\2021_09_13'
        means = []
        stds = []
        data = [i for i in x.T]
        fig7, ax7 = plt.subplots()
        ax7.boxplot(data)
        plt.show()
        self.mg.write_metrics_csv(x, y, path, '2021_09_13')
        # Classify users into fall risk categories
        # Split input data into test and train groups
        x_train, x_test, y_train, y_test = self._generate_test_train_groups(x, y)
        print(self.rc.cross_validate(x, y))
        print('****####****####****####****####****####****####****####****####****')
        # Fit model to training data
        self.rc.train_model(x_train, y_train)
        # Make predictions on the test data
        y_predictions = self.rc.make_prediction(x_test)
        # Compare predictions to test
        return self.rc.create_classification_report(y_test, y_predictions)

    def _build_datasets(self, dataset_info):
        # Read in all builder modules
        metric_modules = [importlib.import_module(module_path).DatasetBuilder()
                          for module_path in self._generate_builder_module_paths()]
        for info in dataset_info:
            name = info['dataset_name']
            mod_list = [mod for mod in metric_modules if mod.get_dataset_name() == name]
            if len(mod_list) == 1:
                mod = mod_list.pop(0)
                self.datasets[name] = mod.build_dataset(info['dataset_path'],
                                                        info['clinical_demo_path'],
                                                        info['segment_dataset'],
                                                        info['epoch_size'])
            else:
                raise ValueError(f'Metric module name not found:{name}')

    def _generate_builder_module_paths(self):
        module_root = 'src.dataset_tools.dataset_builders.builder_instances.'
        module_names = glob.glob(os.path.join(ROOT_DIR, 'src', 'dataset_tools',
                                              'dataset_builders', 'builder_instances',
                                              '*_dataset_builder.py'), recursive=True)
        module_names = [os.path.splitext(os.path.basename(mod_name))[0] for
                        mod_name in module_names]
        module_names = [module_root + mod_name for mod_name in module_names]
        return module_names

    def _preprocess_data(self):
        for name, dataset in self.datasets.items():
            for user_data in dataset.get_dataset():
                # Filter the data
                self._apply_lp_filter(user_data)
                # self.apply_kalman_filter()
                # Remove effects of gravity in vertical axis
                # self._unbias_axes(user_data)

    def _apply_lp_filter(self, user_data):
        imu_data = user_data.get_imu_data()[IMUDataFilterType.RAW]
        v_acc_data = imu_data.get_acc_axis_data('vertical')
        ml_acc_data = imu_data.get_acc_axis_data('mediolateral')
        ap_acc_data = imu_data.get_acc_axis_data('anteroposterior')
        yaw_acc_data = imu_data.get_gyr_axis_data('yaw')
        pitch_acc_data = imu_data.get_gyr_axis_data('pitch')
        roll_acc_data = imu_data.get_gyr_axis_data('roll')
        gyr_data = [yaw_acc_data, pitch_acc_data, roll_acc_data]
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        lpf_data_all_axis = []
        # lpf_data_all_axis.append(
        #     self.filter.apply_lpass_filter(v_acc_data, 0.045))
        # lpf_data_all_axis.append(
        #     self.filter.apply_lpass_filter(v_acc_data, 0.1, samp_freq))
        # lpf_data_all_axis.append(
        #     self.filter.apply_lpass_filter(ml_acc_data, 0.035, samp_freq))
        # lpf_data_all_axis.append(
        #     self.filter.apply_lpass_filter(ap_acc_data, 0.07, samp_freq))
        # for ax in gyr_data:
        #     lpf_data_all_axis.append(
        #         self.filter.apply_lpass_filter(ax, 0.05, samp_freq))
        lpf_data_all_axis.append(
            self.filter.apply_lpass_filter(v_acc_data, 2, samp_freq))
        lpf_data_all_axis.append(
            self.filter.apply_lpass_filter(ml_acc_data, 2, samp_freq))
        lpf_data_all_axis.append(
            self.filter.apply_lpass_filter(ap_acc_data, 2, samp_freq))
        for ax in gyr_data:
            lpf_data_all_axis.append(
                self.filter.apply_lpass_filter(ax, 2, samp_freq))
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis,
                                                        samp_freq)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data, sampling_freq):
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
        time = np.linspace(0, len(v_acc_data) / int(sampling_freq),
                           len(v_acc_data))
        return IMUData(v_acc_data, ml_acc_data, ap_acc_data,
                       yaw_gyr_data, pitch_gyr_data, roll_gyr_data, time)

    def _unbias_axes(self, user_data):
        for imu_filt_type, imu_data in user_data.get_imu_data().items():
            v_acc_data = imu_data.get_acc_axis_data('vertical')
            v_acc_data = np.array(
                [x - v_acc_data.mean() for x in v_acc_data])
            imu_data.v_acc_data = v_acc_data
            ml_acc_data = imu_data.get_acc_axis_data('mediolateral')
            ml_acc_data = np.array(
                [x - ml_acc_data.mean() for x in ml_acc_data])
            imu_data.ml_acc_data = ml_acc_data
            ap_acc_data = imu_data.get_acc_axis_data('anteroposterior')
            ap_acc_data = np.array(
                [x - ap_acc_data.mean() for x in ap_acc_data])
            imu_data.ap_acc_data = ap_acc_data

    def generate_risk_metrics(self, input_metric_names, scale_metrics=True):
        # Separate datasets into fallers and nonfallers
        faller_dataset = []
        non_fallers_dataset = []
        for name, dataset in self.datasets.items():
            faller_dataset.append(dataset.get_data_by_faller_status(True))
            non_fallers_dataset.append(dataset.get_data_by_faller_status(False))
        # Flatten the list of datasets
        faller_dataset = [user_data for data in faller_dataset for user_data in
                          data]
        non_fallers_dataset = [user_data for data in non_fallers_dataset for user_data in
                          data]
        # Generate metrics
        faller_metrics, faller_status = self.mg.generate_metrics(faller_dataset,
                                                                 input_metric_names)
        non_faller_metrics, non_faller_status = self.mg.generate_metrics(non_fallers_dataset,
                                                                         input_metric_names)
        # Transform the train and test input metrics
        x = faller_metrics + non_faller_metrics
        y = faller_status + non_faller_status
        if scale_metrics:
            x = self.rc.scale_input_data(x)
        return x, y

    def assess_model_accuracy(self):
        # Preprocess the data
        self._preprocess_data()
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

    def _call_metric_generator(self, ltmm_dataset):
        faller_status = []
        dataset_metrics = []
        for ltmm_data in ltmm_dataset:
            faller_status.append(int(ltmm_data.get_faller_status()))
            dataset_metrics.append(self._derive_metrics(ltmm_data,
                                                        input_metric_names))
        return list(dataset_metrics), list(faller_status)

    def apply_kalman_filter(self):
        for ltmm_data in self.ltmm_dataset.get_dataset()[:2]:
            v_y_ax_data = ltmm_data.get_axis_acc_data('vertical')
            ml_x_ax_data = ltmm_data.get_axis_acc_data('mediolateral')
            ap_z_ax_data = ltmm_data.get_axis_acc_data('anteroposterior')
            sampling_rate = ltmm_data.get_sampling_frequency()
            kf_x_ml, kf_y_v, kf_z_ap, kf_ub_y_v = self.filter.apply_kalman_filter(ml_x_ax_data,
                                                                                  v_y_ax_data,
                                                                                  ap_z_ax_data,
                                                                                  sampling_rate)
            ltmm_data.get_data()[:, 0] = kf_ub_y_v
            ltmm_data.get_data()[:, 1] = kf_x_ml
            ltmm_data.get_data()[:, 2] = kf_z_ap
            print('kf')

    def _compare_prediction_to_test(self, y_predition, y_test):
        comparison = np.array(np.array(y_predition) == np.array(y_predition),
                              dtype=int)
        return sum(comparison)/len(comparison)

    def _generate_test_train_groups(self, x, y):
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(x, y)
        return x_train, x_test, y_train, y_test


def main():
    # dataset_info: [{dataset_name: DatasetName, dataset_path: path, clin: clin_path, segment_data: bool, epoch: float, mod: MOD}]
    st = time.time()
    # ltmm_dataset_name = 'LTMM'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\small_LTMMD'
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    # report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    input_metric_names = tuple([MetricNames.AUTOCORRELATION,
                                MetricNames.FAST_FOURIER_TRANSFORM,
                                MetricNames.MEAN,
                                MetricNames.ROOT_MEAN_SQUARE,
                                MetricNames.STANDARD_DEVIATION,
                                MetricNames.SIGNAL_ENERGY,
                                MetricNames.COEFFICIENT_OF_VARIANCE,
                                MetricNames.ZERO_CROSSING,
                                MetricNames.SIGNAL_MAGNITUDE_AREA,
                                MetricNames.GAIT_SPEED_ESTIMATOR])
    dataset_info = [{'dataset_name': DatasetNames.LTMM,
                     'dataset_path': ltmm_dataset_path,
                     'clinical_demo_path': clinical_demo_path,
                     'segment_dataset': True,
                     'epoch_size': 8.0}]
    fra = FallRiskAssessment(LightGBMRiskClassifier({}))
    print(fra.perform_risk_assessment(dataset_info, input_metric_names))


    # cv_results = ltmm_ra.cross_validate_model()
    # print(cv_results)

    ft = time.time() - st
    print('##############################################################')
    print(ft)
    print('##############################################################')



if __name__ == '__main__':
    main()
