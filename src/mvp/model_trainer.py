import os
import time
import numpy as np
from typing import Tuple
import joblib


from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


class ModelTrainer:
    def __init__(self, data_path, clinical_demo_path, output_path):
        self.data_path = data_path
        self.clinical_demo_path = clinical_demo_path
        self.output_path = output_path
        self.filter = MotionFilters()
        self.rc = LightGBMRiskClassifier({})
        # self.rc = KNNRiskClassifier({})

    def create_risk_model(self, input_metric_names: Tuple[MetricNames], model_name, scaler_name):
        dataset = self.load_data()
        self.preprocess_data(dataset)
        input_metrics: InputMetrics = self.gen_input_metrics(dataset, input_metric_names)
        self.train_model(input_metrics, input_metric_names)
        model_path = self.export_model(model_name, scaler_name)
        model = self.import_model(model_path)
        print('YUSSS')

    def load_data(self):
        db = DatasetBuilder()
        return db.build_dataset(self.data_path, self.clinical_demo_path, True, 8.0)

    def preprocess_data(self, dataset):
        for user_data in dataset.get_dataset():
            # Filter the data
            self.apply_lp_filter(user_data)

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

    def gen_input_metrics(self, dataset, input_metric_names):
        mg = MetricGenerator()
        input_metrics = mg.generate_metrics(dataset.get_dataset(), input_metric_names)
        return self.rc.scale_input_data(input_metrics)

    def train_model(self, input_metrics, input_metric_names):
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        self.rc.train_model(x, y, metric_names = input_metric_names)

    def export_model(self, model_name, scaler_name):
        model_path = os.path.join(self.output_path, model_name)
        scaler_path = os.path.join(self.output_path, scaler_name)
        # self.rc.model.save_model(model_path)
        joblib.dump(self.rc.get_model(), model_path)
        joblib.dump(self.rc.get_scaler(), scaler_path)
        return model_path

    def import_model(self, model_path):
        model = joblib.load(model_path)
        return model


def main():
    # VM paths
    ltmm_dataset_path = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/LabWalks'
    clinical_demo_path = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    output_path = ''

    # Desktop paths
    # ltmm_dataset_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    # clinical_demo_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    # output_path = r'C:\Users\gsass\Documents\Fall Project Master\fafra_testing\fibion\risk_models'

    # Laptop paths
    # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
    # clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
    # output_path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\fibion\risk_models'

    mt = ModelTrainer(ltmm_dataset_path, clinical_demo_path, output_path)
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
    # input_metric_names = tuple([MetricNames.AUTOCORRELATION,
    #                             MetricNames.FAST_FOURIER_TRANSFORM,
    #                             MetricNames.MEAN,
    #                             MetricNames.ROOT_MEAN_SQUARE,
    #                             MetricNames.STANDARD_DEVIATION,
    #                             MetricNames.SIGNAL_ENERGY,
    #                             MetricNames.COEFFICIENT_OF_VARIANCE,
    #                             MetricNames.ZERO_CROSSING,
    #                             MetricNames.SIGNAL_MAGNITUDE_AREA])
    model_name = 'lgbm_fafra_rcm_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
    scaler_name = 'lgbm_fafra_scaler_' + time.strftime("%Y%m%d-%H%M%S") + '.bin'
    mt.create_risk_model(input_metric_names, model_name, scaler_name)

if __name__ == '__main__':
    main()
