import numpy as np
import time
import os
import glob
import importlib
import random
import json
from typing import Tuple, Dict, List, Any
from definitions import ROOT_DIR
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import sys,getopt

from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.dataset_builders.dataset_builder import DatasetBuilder

from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.risk_classification.validation.cross_validator import CrossValidator
from src.risk_classification.validation.input_metric_validator import InputMetricValidator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.risk_classification.risk_classifiers.classifier import Classifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.motion_analysis.attitude_estimation.attitude_estimator import AttitudeEstimator

from src.risk_classification.validation.classifier_evaluator import ClassifierEvaluator
from src.risk_classification.validation.classifier_metrics import ClassifierMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric


class FallRiskAssessment:
    def __init__(self, risk_classifier: Classifier):
        # Required input parameters
        self.dataset_builders: Dict[str: DatasetBuilder] = {}
        self.datasets: Dict[DatasetNames: Dataset] = {}
        self.rc: Classifier = risk_classifier
        self.filter = MotionFilters()
        self.rc_viz = ClassificationVisualizer()
        #self.m_viz = MetricViz()
        self.mg = MetricGenerator()
        self.cv = CrossValidator()
        self.scaler: StandardScaler = StandardScaler()
        self.att_est: AttitudeEstimator = AttitudeEstimator()

    def perform_risk_assessment(self, dataset_info: List[Dict[str, Any]],
                                input_metric_names: Tuple[MetricNames],
                                output_path=None):
        # input_metrics
        # Build datasets from given names
        self._build_datasets(dataset_info)
        # Preprocess imu data
        self._preprocess_data()
        # Derive risk metrics
        random.shuffle(self.datasets[DatasetNames.LTMM].get_dataset())
        input_metrics: InputMetrics = self.generate_risk_metrics(input_metric_names)
        print(input_metrics)
        # Using canonical notation for input vectors, x and y
        # x = input_metrics.get_metrics()
        # y = input_metrics.get_labels()
        # self.m_viz.violin_plot_metrics(x, y)
        # Scale input metrics
        input_metrics = self.rc.scale_input_data(input_metrics)
        self.inputM=input_metrics
        x = input_metrics.get_metrics()
        y = input_metrics.get_labels()
        self.rc_viz.violin_plot_metrics(x, y)
        self.rc_viz.corr_linkage(input_metrics)
        # Classify users into fall risk categories
        # Split input data into test and train groups
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(
            input_metrics)
        cv_x, names = input_metrics.get_metric_matrix()
        cv_results = self.rc.cross_validate(cv_x, y)
        print("cross val done",cv_results)
        # Fit model to training data
        self.rc.train_model(x_train, y_train, metric_names=input_metric_names)
        # Make predictions on the test data
        y_predictions = self.rc.make_prediction(x_test)
        y_predictions = [int(i) for i in y_predictions]
        class_report = self.rc.create_classification_report(y_test, y_predictions)
        
        # input_validator= InputMetricValidator()
        # input_validator.perform_partial_dependence_plot_lightGBM(self.rc,input_metrics,y)
        #input_validator.perform_partial_dependence_plot_sklearn(self.rc,input_metrics,y)
        #input_validator.perform_shap_values(self.rc,input_metrics)
        #input_validator.perform_permutation_feature_importance(self.rc,input_metrics,y)

        if output_path:
            self._write_results(output_path, x, x_train, x_test, y_train, y_test,
                       y_predictions, cv_results, class_report)

        print(cv_results)
        print(class_report)

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
                self.apply_lp_filter(user_data)
                #self.att_est.estimate_attitude(user_data)
                # self.apply_kalman_filter()
                # Remove effects of gravity in vertical axis
                # self._unbias_axes(user_data)

    def apply_lp_filter(self, user_data):
        imu_data: IMUData = user_data.get_imu_data()[IMUDataFilterType.RAW]
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        act_code = imu_data.get_activity_code()
        act_des = imu_data.get_activity_description()
        all_raw_data = imu_data.get_all_data()
        lpf_data_all_axis = []
        for data in all_raw_data:
            lpf_data_all_axis.append(
                self.filter.apply_lpass_filter(data, 2, samp_freq))
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis,
                                                        samp_freq, act_code, act_des)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data, sampling_freq, act_code, act_des):
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
        time = np.linspace(0, len(v_acc_data) / int(sampling_freq),
                           len(v_acc_data))
        return IMUData(act_code, act_des, v_acc_data, ml_acc_data, ap_acc_data,
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

    def generate_risk_metrics(self, input_metric_names):
        # Separate datasets into fallers and nonfallers
        user_data = []
        for name, dataset in self.datasets.items():
            user_data.extend(dataset.get_dataset())
        return self.mg.generate_metrics(user_data, input_metric_names)

    def assess_model_accuracy(self):
        # Preprocess the data
        self._preprocess_data()
        # Generate risk metrics
        x, y = self.generate_risk_metrics()
        # Split input data into test and train groups
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(x, y)
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

    def _write_results(self, output_path, x, x_train, x_test, y_train, y_test,
                       y_predictions, cv_results, class_report):
        # Create results_dir folder with date, time, uuid
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_name = self.rc.get_name()
        results_dir = 'results_' + model_name + '_' + timestr
        results_path = os.path.join(output_path, results_dir)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # Format inputs to write to file
        input_metrics_labels = {
            'metrics':
                [
                    {'metric': 'x_train',
                     'value': x_train.tolist()
                     },
                    {'metric': 'x_test',
                     'value': x_test.tolist()
                     }
                ],
            'labels':
                [
                    {'label': 'y_train',
                     'value': y_train
                     },
                    {'label': 'y_test',
                     'value': y_test
                     }
                ]
        }
        for name, value in x.items():
            input_metrics_labels[name]: value
        ml_filename = os.path.join(results_path, 'input_metrics_labels.json')
        # Format model params to write to file
        params = self.rc.get_params()
        params['model_name'] = model_name
        params_filename = os.path.join(results_path, 'model_params.json')
        # Format results to write to file
        cv_results['fit_time'] = cv_results['fit_time'].tolist()
        cv_results['score_time'] = cv_results['score_time'].tolist()
        cv_results['test_score'] = cv_results['test_score'].tolist()
        results = {
            'predictions': y_predictions,
            'cv_results': cv_results,
            'classification_report': class_report
        }
        results_filename = os.path.join(results_path, 'results.json')
        # Write outputs to JSON file
        with open(ml_filename, 'w') as mlf:
            json.dump(input_metrics_labels, mlf)
        with open(params_filename, 'w') as pf:
            json.dump(params, pf)
        with open(results_filename, 'w') as rf:
            json.dump(results, rf)
    def classifier_evaluator(self,output_path):
        cl_ev = ClassifierEvaluator()
        eval_metrics = [ClassifierMetrics.PDP_KNN,ClassifierMetrics.LIME]
        # classifiers = [KNNRiskClassifier(), LightGBMRiskClassifier({}),
        #                SVMRiskClassifier()]
        classifiers = [self.rc]
        input_metrics = self.inputM
        #metric_name = MetricNames.EXAMPLE
        #input_metric = InputMetric(metric_name, np.array([]))
        #input_metrics.set_metric(MetricNames.EXAMPLE, input_metric)
        #y = np.array([])
        #input_metrics.set_labels(y)
        #output_path = r'F:\long-term-movement-monitoring-database-1.0.0\output_dir'
        cl_ev.run_models_evaluation(eval_metrics, classifiers, input_metrics, output_path)

def main():
    # dataset_info: [{dataset_name: DatasetName, dataset_path: path, clin: clin_path, segment_data: bool, epoch: float, mod: MOD}]
    argumentList=sys.argv[1:]
    opts='io:'
    long_opts=['input_add','output_add']
    argument,values=getopt.getopt(argumentList,opts,long_opts)
    for cur_Arg,cur_val in argument:
        if cur_Arg in('-o','--output_add'):
            output_dir=cur_val
    st = time.time()
    # ltmm_dataset_name = 'LTMM'

    # Desktop paths
    ltmm_dataset_path = r'F:\long-term-movement-monitoring-database-1.0.0\long-term-movement-monitoring-database-1.0.0\LabWalks'
    # ltmm_dataset_path = r'C:\Users\gsass\Documents\fafra\datasets\LTMM\LTMM_database-1.0.0'
    clinical_demo_path = r'F:\long-term-movement-monitoring-database-1.0.0\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'

    # Laptop paths
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\small_LTMMD'
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    # clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    # report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'

    #output_dir = r'F:\long-term-movement-monitoring-database-1.0.0\output_dir'
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
    #fra = FallRiskAssessment(KNNRiskClassifier())
    light_gbm_classifier = LightGBMRiskClassifier({})
    knn_classifier = KNNRiskClassifier()
    svm_classifier = SVMRiskClassifier()
    fra = FallRiskAssessment(knn_classifier)
    fra.perform_risk_assessment(dataset_info, input_metric_names, output_dir)
    fra.classifier_evaluator(output_dir)


    # cv_results = ltmm_ra.cross_validate_model()
    # print(cv_results)

    ft = time.time() - st
    print('##############################################################')
    print('Runtime: ' + str(ft))
    print('##############################################################')



if __name__ == '__main__':
    main()
