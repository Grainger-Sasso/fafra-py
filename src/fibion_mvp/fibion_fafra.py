import time
import numpy as np
import pandas as pd
import calendar
import bisect
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import parser, tz
import joblib

from src.motion_analysis.filters.motion_filters import MotionFilters
from src.fibion_mvp.fibion_dataset_builder import FibionDatasetBuilder
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.motion_analysis.gait_analysis.gait_analyzer_v2 import GaitAnalyzerV2
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric


class FibionFaFRA:
    def __init__(self, dataset_path, activity_path, demo_data, timezone=tz.gettz("America/New_York")):
        self.dataset_path = dataset_path
        self.activity_path = activity_path
        self.dataset = self.load_dataset(self.dataset_path, demo_data)
        self.activity_data = self.load_activity_data(activity_path, timezone)
        self.filter = MotionFilters()
        self.mg = MetricGenerator()
        self.gse = GaitAnalyzerV2()
        # Laptop RC Paths
        # self.rc_path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\fibion\risk_models\lgbm_fafra_rcm_20220625-140624.pkl'
        # self.rc_scaler_path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\fibion\risk_models\lgbm_fafra_scaler_20220625-140624.bin'
        # Desktop RC paths
        # self.rc_path = r'C:\Users\gsass\Documents\Fall Project Master\fafra_testing\fibion\risk_models\lgbm_fafra_rcm_20220706-112631.pkl'
        # self.rc_scaler_path = r'C:\Users\gsass\Documents\Fall Project Master\fafra_testing\fibion\risk_models\lgbm_fafra_scaler_20220706-112631.bin'
        # VM RC paths
        self.rc_path = '/home/grainger/Desktop/risk_classifiers/lgbm_fafra_rcm_20220706-112631.pkl'
        self.rc_scaler_path = '/home/grainger/Desktop/risk_classifiers/lgbm_fafra_scaler_20220706-112631.bin'
        self.rc = LightGBMRiskClassifier({})

    def load_dataset(self, dataset_path, demo_data):
        fdb = FibionDatasetBuilder()
        ds = fdb.build_dataset(dataset_path, demo_data, '')
        return ds

    def load_activity_data(self, activity_path, timezone):
        act_data = pd.read_csv(activity_path)
        # Add column with epoch time
        epochs = []
        local_times = []
        valid_data_ixs = []
        # Get each row, get the epoch from that row and whether there is data (no data time = 15)
        for ix, row in act_data.iterrows():
            date = parser.parse(row['utc']).replace(tzinfo=tz.gettz('UTC'))
            local_time = date.astimezone(timezone)
            epochs.append(local_time.timestamp())
            local_times.append(local_time)
            if row[' general/nodata/time'] != 15.0:
                valid_data_ixs.append(ix)
        act_data['epoch'] = epochs
        act_data['converted_local_time'] = local_times
        return act_data

    def perform_risk_analysis(self, input_metric_names=tuple(MetricNames.get_all_enum_entries())):
        t0 = time.time()
        # Preprocess subject data (low-pass filtering)
        self.preprocess_data()
        print(f'{time.time() - t0}')
        # Estimate subject gait speed
        gait_speed = self.estimate_gait_speed(self.dataset)
        # Get subject step count from activity chart
        step_count = self.get_ac_step_count()
        # Estimate subject activity levels
        act_levels = self.estimate_activity_levels()
        # Get sleep disturbances from activity chart
        sleep_dis = self.get_ac_sleep_disturb()
        # Estimate subject fall risk
        fall_risk_score = self.estimate_fall_risk(input_metric_names, gait_speed)
        # Evaluate user fall risk status
        # Evaluate faller risk levels
        # Build risk report
        self.build_risk_report(fall_risk_score)

    def estimate_gait_speed(self, dataset):
        # Initialize gse variables
        gait_speed_estimates = []
        ga = GaitAnalyzerV2()
        # GSE params
        hpf = False
        max_com_v_delta = 0.14
        plot_gait_cycles = False
        # For every epoch in the data
        for user_data in self.dataset.get_dataset():
            # If the epoch has walking bouts in it according to Fibion data
            if self._check_walking_bout(user_data):
                # Run the epoch through the GSE
                gait_speed, fig, gait_params = self.gse.estimate_gait_speed(
                    user_data, hpf, max_com_v_delta, plot_gait_cycles)
                if not np.isnan(gait_speed):
                    gait_speed_estimates.append(gait_speed)
                # if fig and not np.isnan(gait_speed):
                #     fig.show()
                plt.close()
        return np.array(gait_speed_estimates).mean()

    def _check_walking_bout(self, user_data):
        walking = False
        t0 = user_data.get_imu_data(IMUDataFilterType.RAW).get_time()[0]
        epoch_ix = bisect.bisect_right(self.activity_data['epoch'], t0) - 1
        if epoch_ix > 0:
            if self.activity_data[' activity/upright_walk/time'][epoch_ix] > 0:
                print(t0)
                print(self.activity_data['epoch'][epoch_ix])
                print(self.activity_data['converted_local_time'][epoch_ix])
                print('')
                walking = True
        return walking

    def calc_gait_speeds_v2(self, dataset: Dataset, eval_percentages,
                            results_location, write_out_estimates=False,
                            hpf=False, max_com_v_delta=0.14,
                            plot_gait_cycles=False):
        # TODO: fix the parameters that control plotting and exporting data, also fix how figures are created
        # Estimate the gait speed for every user/trial in dataset
        truth_comparisons = []
        all_gait_params = []
        ga = GaitAnalyzerV2()
        if write_out_estimates:
            # Generate the directories based on evaluation percentages
            percentage_dirs = self.generate_percentage_dirs(eval_percentages, results_location)
        for user_data in dataset.get_dataset():
            user_id = user_data.get_clinical_demo_data().get_id()
            trial = user_data.get_clinical_demo_data().get_trial()
            gait_speed, fig, gait_params = ga.estimate_gait_speed(user_data, hpf, max_com_v_delta,
                                                plot_gait_cycles)
            all_gait_params.append({'user_data': user_data, 'gait_params': gait_params})
            result = {'id': user_id, 'trial': trial, 'gait_speed': gait_speed,
                      'user_data': user_data}
            if user_id in self.subj_gs_truth.keys():
                comparison = self.compare_gs_to_truth_val(result)
                truth_comparisons.append(comparison)
                if write_out_estimates:
                    self.write_out_gs2_comparison_results(comparison, fig,
                                                          eval_percentages,
                                                          percentage_dirs)
            plt.close()
        return truth_comparisons, all_gait_params

    def get_ac_sleep_disturb(self):
        pass

    def estimate_activity_levels(self):
        pass

    def get_ac_step_count(self):
        return self.activity_data[' activity/steps2/count'].sum()

    def estimate_fall_risk(self, input_metric_names, gait_speed):
        # Import risk model
        model, scaler = self.import_model()
        self.rc.set_model(model)
        self.rc.set_scaler(scaler)
        # Compute input metrics
        input_metrics: InputMetrics = self.generate_risk_metrics(
            input_metric_names)
        self.take_epoch_average(input_metrics)
        # Insert gait speed metrics to metrics
        metrics, names = self.format_input_metrics_scaling(input_metrics, gait_speed)
        # Scale input metrics via scaler transformation
        metrics = self.rc.scaler.transform(metrics)
        # Format input data for model prediction
        # Make fall risk prediction on trained model
        return self.rc.make_prediction(metrics)[0]


    def format_input_metrics_scaling(self, input_metrics, gait_speed):
        metrics, names = input_metrics.get_metric_matrix()
        metrics = metrics.tolist()
        metrics.insert(3, gait_speed)
        metrics = np.array(metrics)
        metrics = np.reshape(metrics, (1, -1))
        names.insert(3, MetricNames.GAIT_SPEED_ESTIMATOR)
        return metrics, names

    def take_epoch_average(self, input_metrics: InputMetrics):
        for name, metric in input_metrics.get_metrics().items():
            metric.value = metric.value.mean()

    def import_model(self):
        model = joblib.load(self.rc_path)
        scaler = joblib.load(self.rc_scaler_path)
        return model, scaler

    def build_risk_report(self, fall_risk_score):
        print(fall_risk_score)

    def generate_risk_metrics(self, input_metric_names):
        # Separate datasets into fallers and nonfallers
        return self.mg.generate_metrics(self.dataset.get_dataset(), input_metric_names)

    def preprocess_data(self):
        for user_data in self.dataset.get_dataset():
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
            lpf_data = self.filter.apply_lpass_filter(data, 2, samp_freq) if data.any() else data
            lpf_data_all_axis.append(lpf_data)
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis,
                                                        samp_freq, act_code,
                                                        act_des)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data, sampling_freq, act_code, act_des):
        # TODO: Finish reformatting the imu data for new data instances after lpf
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


def main():
    # Grainger VM paths
    dataset_path = '/home/grainger/Desktop/datasets/fibion/io_test_data/bin'
    activity_path = '/home/grainger/Desktop/datasets/fibion/io_test_data/activity/fibion_test_activity_04_10_2022.csv'

    # Grainger desktop paths
    # dataset_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin'
    # activity_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\activity\fibion_test_activity_04_10_2022.csv'

    # Grainger laptop paths
    # dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\bin'
    # activity_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\csv\export_2022-04-11T01_00_00.000000Z.csv'
    demo_data = {'user_height': 1.88}
    # activity_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\csv\2022-04-12_activity_file.csv'
    fib_fafra = FibionFaFRA(dataset_path, activity_path, demo_data)
    # input_metric_names = tuple([MetricNames.AUTOCORRELATION,
    #                             MetricNames.FAST_FOURIER_TRANSFORM,
    #                             MetricNames.MEAN,
    #                             MetricNames.ROOT_MEAN_SQUARE,
    #                             MetricNames.STANDARD_DEVIATION,
    #                             MetricNames.SIGNAL_ENERGY,
    #                             MetricNames.COEFFICIENT_OF_VARIANCE,
    #                             MetricNames.ZERO_CROSSING,
    #                             MetricNames.SIGNAL_MAGNITUDE_AREA,
    #                             MetricNames.GAIT_SPEED_ESTIMATOR])
    input_metric_names = tuple([MetricNames.AUTOCORRELATION,
                                MetricNames.FAST_FOURIER_TRANSFORM,
                                MetricNames.MEAN,
                                MetricNames.ROOT_MEAN_SQUARE,
                                MetricNames.STANDARD_DEVIATION,
                                MetricNames.SIGNAL_ENERGY,
                                MetricNames.COEFFICIENT_OF_VARIANCE,
                                MetricNames.ZERO_CROSSING,
                                MetricNames.SIGNAL_MAGNITUDE_AREA])
    fib_fafra.perform_risk_analysis(input_metric_names)


if __name__ == '__main__':
    main()

