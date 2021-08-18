import numpy as np
import time
import os
import glob
import importlib
from typing import Tuple, Dict, List, Any
from definitions import ROOT_DIR

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


class FallRiskAssessment:
    def __init__(self, risk_classifier):
        # Required input parameters
        self.dataset_builders: Dict[str: DatasetBuilder] = {}
        self.datasets: Dict[DatasetNames: Dataset] = {}
        self.rc = risk_classifier
        self.filter = MotionFilters()
        self.rc_viz = ClassificationVisualizer()
        self.mg = MetricGenerator()
        self.cv = CrossValidator()

    def perform_risk_assessment(self, dataset_info: List[Dict[str, Any]],
                                input_metric_names: Tuple[MetricNames]):
        # input_metrics
        # Build datasets from given names
        self._build_datasets(dataset_info)
        # Preprocess imu data
        self._preprocess_data()
        # Derive risk metrics
        x, y = self.generate_risk_metrics(input_metric_names)
        # Classify users into fall risk categories
        # Split input data into test and train groups
        x_train, x_test, y_train, y_test = self._generate_test_train_groups(x, y)
        # Fit model to training data
        self.rc.fit_model(x_train, y_train)
        # Make predictions on the test data
        y_predictions = self.rc.make_prediction(x_test)
        # Compare predictions to test
        return self.rc.create_classification_report(y_test, y_predictions)

    def _build_datasets(self, dataset_info):
        # Read in all builder modules
        module_paths = self._generate_builder_module_paths()
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
                self._unbias_vert_axis(user_data)

    def _apply_lp_filter(self, user_data):
        raw_data = user_data.get_imu_data()[IMUDataFilterType.RAW].get_all_data()
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        lpf_data_all_axis = []
        for axis in raw_data:
            lpf_data_all_axis.append(self.filter.apply_lpass_filter(axis,
                                                                    samp_freq))
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data):
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
        return IMUData(v_acc_data, ml_acc_data, ap_acc_data,
                       yaw_gyr_data, pitch_gyr_data, roll_gyr_data)

    def _unbias_vert_axis(self, user_data):
        for imu_filt_type, imu_data in user_data.get_imu_data().items():
            v_acc_data = imu_data.get_acc_axis_data('vertical')
            v_acc_data = np.array([x - 1.0 for x in v_acc_data])
            imu_data.v_acc_data = v_acc_data

    def generate_risk_metrics(self, input_metric_names, scale_data=True):
        # Separate datasets into fallers and nonfallers
        faller_dataset = []
        non_fallers_dataset = []
        for name, dataset in self.datasets.items():
            faller_dataset.append(dataset.get_data_by_faller_status(True))
            non_fallers_dataset.append(dataset.get_data_by_faller_status(False))
        # Generate metrics
        faller_metrics, faller_status = self.mg.generate_metrics(faller_dataset,
                                                                 input_metric_names)
        non_faller_metrics, non_faller_status = self.mg.generate_metrics(non_fallers_dataset,
                                                                         input_metric_names)
        # Transform the train and test input metrics
        x = faller_metrics + non_faller_metrics
        y = faller_status + non_faller_status
        if scale_data:
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
    ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    # report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
    input_metric_names = tuple([MetricNames.GAIT_SPEED_ESTIMATOR])
    dataset_info = [{'dataset_name': DatasetNames.LTMM,
                     'dataset_path': ltmm_dataset_path,
                     'clinical_demo_path': clinical_demo_path,
                     'segment_dataset': True,
                     'epoch_size': 10.0}]
    fra = FallRiskAssessment('')
    print(fra.perform_risk_assessment(dataset_info, input_metric_names))


    # cv_results = ltmm_ra.cross_validate_model()
    # print(cv_results)

    ft = time.time() - st
    print('##############################################################')
    print(ft)
    print('##############################################################')



if __name__ == '__main__':
    main()
