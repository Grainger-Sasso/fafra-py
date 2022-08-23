import os
import time
import numpy as np
import psutil
import gc
import pandas as pd
import glob
import wfdb
import json
import calendar
import bisect
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import parser, tz
from typing import List
from skdh import Pipeline
from skdh.gait import Gait
from skdh.sleep import Sleep
from skdh.activity import ActivityLevelClassification
import joblib

from src.motion_analysis.filters.motion_filters import MotionFilters
from src.fibion_mvp.fibion_dataset_builder import FibionDatasetBuilder
from src.fibion_mvp.mbientlab_dataset_builder import MbientlabDatasetBuilder
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
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData
from src.fibion_mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner


class FaFRA_SKDH:
    def __init__(self, dataset_path, activity_path, demo_data, output_path, timezone=tz.gettz("America/New_York")):
        self.dataset_path = dataset_path
        self.activity_path = activity_path
        self.output_path = output_path
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

    def perform_risk_assessment(self, data_path, demographic_data, skdh_metric_path, custom_metric_path, model_path, scaler_path, imu_data_type='mbientlab'):
        # Load in the accelerometer data
        ds = self.load_dataset(data_path, demographic_data, imu_data_type)
        # Calculate day ends
        # day_ends = self.get_day_ends(ds)
        day_ends = day_ends = np.array([[0, 3836477], [3836477, 7607840]])
        # Generate custom metrics and SKDH metrics on the user data
        metric_gen = FaFRAMetricGenerator()
        path, metrics = metric_gen.generate_input_metrics(ds, day_ends, skdh_metric_path, custom_metric_path)
        # metrics = {"metrics": [{"sma": 5684.0631103515625, "cov": 0.07972064126298378, "std": 0.08061715722448112, "mean": 1.010986328125, "se": 5564.046325683594, "rms": 1.0142822265625, "PARAM:gait speed: mean": 0.8239671088093871, "PARAM:gait speed: std": 0.044765261012231314, "BOUTPARAM:gait symmetry index: mean": 0.7676332028836402, "BOUTPARAM:gait symmetry index: std": 0.003438609506065257, "PARAM:cadence: mean": 104.72948932441702, "PARAM:cadence: std": 4.186624922070641, "Bout Steps: mean": 38.004464285714285, "Bout Steps: std": 2.05752240267269, "Bout Duration: mean": 21.683035714285715, "Bout Duration: std": 1.089276566120836, "Bout N: mean": 1.0334821428571428, "Bout N: std": 0.060515364784490884, "Bout Starts: mean": 13.506696428571429, "Bout Starts: std": 2.72319141530209, "PARAM:stride time: mean": 1.15470404944687, "PARAM:stride time: std": 0.03649495933789282, "PARAM:stride time asymmetry: mean": -0.0014657286428059761, "PARAM:stride time asymmetry: std": 0.022568032803646724, "PARAM:stance time: mean": 0.7358104718743173, "PARAM:stance time: std": 0.028221579403790803, "PARAM:stance time asymmetry: mean": -2.373408837353817e-05, "PARAM:stance time asymmetry: std": 0.02615877675696673, "PARAM:swing time: mean": 0.42190154438280836, "PARAM:swing time: std": 0.018620921895707403, "PARAM:swing time asymmetry: mean": 8.592450484146373e-05, "PARAM:swing time asymmetry: std": 0.023034650217666423, "PARAM:step time: mean": 0.5787774925345458, "PARAM:step time: std": 0.023718435414781542, "PARAM:step time asymmetry: mean": -0.0004269219979692915, "PARAM:step time asymmetry: std": 0.026327685331106427, "PARAM:initial double support: mean": 0.15520912106727264, "PARAM:initial double support: std": 0.009487038182593386, "PARAM:initial double support asymmetry: mean": 0.00024082366732392177, "PARAM:initial double support asymmetry: std": 0.012191312369886536, "PARAM:terminal double support: mean": 0.15533371570027718, "PARAM:terminal double support: std": 0.008611994210791464, "PARAM:terminal double support asymmetry: mean": -0.00020961494645705165, "PARAM:terminal double support asymmetry: std": 0.011579802116538779, "PARAM:double support: mean": 0.3104266077332305, "PARAM:double support: std": 0.01328365214579084, "PARAM:double support asymmetry: mean": 0.0001978946998697179, "PARAM:double support asymmetry: std": 0.011780234377643528, "PARAM:single support: mean": 0.4236846005015926, "PARAM:single support: std": 0.02114958420906941, "PARAM:single support asymmetry: mean": -0.0008344316442960612, "PARAM:single support asymmetry: std": 0.02354894505532213, "PARAM:step length: mean": 0.4741583681952849, "PARAM:step length: std": 0.046668206697445876, "PARAM:step length asymmetry: mean": -0.0034886191276315586, "PARAM:step length asymmetry: std": 0.07336069895530065, "PARAM:stride length: mean": 0.9485873159068022, "PARAM:stride length: std": 0.05580781054130711, "PARAM:stride length asymmetry: mean": -0.006535574290192136, "PARAM:stride length asymmetry: std": 0.04727984795524197, "PARAM:gait speed asymmetry: mean": -0.004234233899815849, "PARAM:gait speed asymmetry: std": 0.03890404491011089, "PARAM:intra-step covariance - V: mean": 0.9785122744370086, "PARAM:intra-step covariance - V: std": 0.02095334087831291, "PARAM:intra-stride covariance - V: mean": 0.9703352801519978, "PARAM:intra-stride covariance - V: std": 0.04280852612274942, "PARAM:harmonic ratio - V: mean": 1.7886702922453732, "PARAM:harmonic ratio - V: std": 0.7764661975991594, "PARAM:stride SPARC: mean": -4.186293910645728, "PARAM:stride SPARC: std": 0.18420241405618507, "BOUTPARAM:phase coordination index: mean": 4.12110230471326, "BOUTPARAM:phase coordination index: std": 0.008082082635380265, "Bout Steps: sum": 36.125, "Bout Duration: sum": 20.625}], "labels": [0, 0, 0, 0, 0, 0, 0, 0]}
        risk_model = self.import_classifier(model_path, scaler_path)
        metrics = self.format_input_metrics_scaling(metrics)
        metrics = risk_model.scaler.transform(metrics)
        pred = risk_model.make_prediction(metrics)[0]
        print(pred)
        # Assess risk levels using risk model
        # TEST COMMIT
        pass

    def load_dataset(self, dataset_path, demo_data, imu_data_type):
        if imu_data_type == 'mbientlab':
            db = MbientlabDatasetBuilder()
            ds = db.build_dataset(dataset_path, demo_data, '')
            print('MbientLab dataset build...')
        elif imu_data_type == 'fibion':
            db = FibionDatasetBuilder()
            ds = db.build_single_user(dataset_path, demo_data)
            ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).time = np.array(
                ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).get_time())
            print('Fibion dataset built...')
        else:
            raise ValueError(f'Incorrect IMU data type provided: {imu_data_type}')
        return ds

    def get_day_ends(self, ds):
        time = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).get_time()
        current_ix = 0
        iter_ix = 0
        day_end_pairs = []
        while iter_ix + 1 <= len(time) - 1:
            if datetime.fromtimestamp(time[iter_ix]).time().hour > datetime.fromtimestamp(
                    time[iter_ix + 1]).time().hour:
                day_end_pairs.append([current_ix, iter_ix])
                current_ix = iter_ix
            iter_ix += 1
        day_end_pairs.append([current_ix, len(time) - 1])
        return day_end_pairs

    def import_classifier(self, model_path, scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        classifier = LightGBMRiskClassifier({})
        classifier.set_model(model)
        classifier.set_scaler(scaler)
        return classifier

    def format_input_metrics_scaling(self, input_metrics):
        metrics = input_metrics.get_metrics()
        new_metrics = []
        for name, metric in metrics.items():
            new_metrics.append(metric)
        metrics = np.array(new_metrics)
        metrics = np.reshape(metrics, (1, -1))
        return metrics


class FaFRAMetricGenerator:
    def __init__(self):
        self.custom_metric_names = tuple(
            [
                MetricNames.SIGNAL_MAGNITUDE_AREA,
                MetricNames.COEFFICIENT_OF_VARIANCE,
                MetricNames.STANDARD_DEVIATION,
                MetricNames.MEAN,
                MetricNames.SIGNAL_ENERGY,
                MetricNames.ROOT_MEAN_SQUARE
            ]
        )
        self.gait_metric_names: List[str] = [
                'PARAM:gait speed',
                'BOUTPARAM:gait symmetry index',
                'PARAM:cadence',
                'Bout Steps',
                'Bout Duration',
                'Bout N',
                'Bout Starts',
                #Additional gait params
                'PARAM:stride time',
                'PARAM:stride time asymmetry',
                'PARAM:stance time',
                'PARAM:stance time asymmetry',
                'PARAM:swing time',
                'PARAM:swing time asymmetry',
                'PARAM:step time',
                'PARAM:step time asymmetry',
                'PARAM:initial double support',
                'PARAM:initial double support asymmetry',
                'PARAM:terminal double support',
                'PARAM:terminal double support asymmetry',
                'PARAM:double support',
                'PARAM:double support asymmetry',
                'PARAM:single support',
                'PARAM:single support asymmetry',
                'PARAM:step length',
                'PARAM:step length asymmetry',
                'PARAM:stride length',
                'PARAM:stride length asymmetry',
                'PARAM:gait speed asymmetry',
                'PARAM:intra-step covariance - V',
                'PARAM:intra-stride covariance - V',
                'PARAM:harmonic ratio - V',
                'PARAM:stride SPARC',
                'BOUTPARAM:phase coordination index',
                'PARAM:intra-step covariance - V',
                'PARAM:intra-stride covariance - V',
                'PARAM:harmonic ratio - V',
                'PARAM:stride SPARC',
                'BOUTPARAM:phase coordination index'
            ]

    def generate_input_metrics(self, ds, day_ends, skdh_output_path, im_path, seg_gait=True, min_gait_dur=30.0):
        pipeline_gen = SKDHPipelineGenerator()
        full_pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        full_pipeline_run = SKDHPipelineRunner(full_pipeline, self.gait_metric_names)
        gait_pipeline = pipeline_gen.generate_gait_pipeline(skdh_output_path)
        gait_pipeline_run = SKDHPipelineRunner(gait_pipeline, self.gait_metric_names)

        self.preprocess_data(ds)
        skdh_input_metrics = self.generate_skdh_metrics(ds, day_ends, full_pipeline_run, False)
        input_metrics = self.initialize_input_metrics(skdh_input_metrics)
        self.export_skdh_results(skdh_input_metrics, skdh_output_path)
        custom_input_metrics: InputMetrics = self.generate_custom_metrics(ds)
        # If the data is to be segmented along walking data
        if seg_gait:
            # Run the gait pipeline
            # Segment the data along the walking bouts
            bout_ixs = self.get_walk_bout_ixs(skdh_input_metrics, ds, min_gait_dur)
            if bout_ixs:
                walk_data = self.get_walk_imu_data(bout_ixs, ds)
                # Create new dataset from the walking data segments
                walk_ds = self.gen_walk_ds(walk_data, ds)
                self.preprocess_data(walk_ds)
                custom_input_metrics: InputMetrics = self.generate_custom_metrics(walk_ds)
                skdh_input_metrics = self.generate_skdh_metrics(walk_ds, day_ends, gait_pipeline_run, True)
            else:
                print('FAILED TO SEGMENT DATA ALONG GAIT BOUTS')
        print(input_metrics)
        print(custom_input_metrics)
        print(skdh_input_metrics)
        print(len(skdh_input_metrics))
        for metric in skdh_input_metrics:
            print(metric)
        input_metrics = self.format_input_metrics(input_metrics,
                                                  custom_input_metrics, skdh_input_metrics)
        print(input_metrics.get_metrics())
        print(len(input_metrics.get_metrics()))
        full_path = self.export_metrics(input_metrics, im_path)
        print(full_path)
        return full_path, input_metrics

    def export_skdh_results(self, results, path):
        result_file_name = 'skdh_results_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        full_path = os.path.join(path, result_file_name)
        new_results = {}
        for name, item in results[0].items():
            new_results[name] = {}
            for nest_name, nest_item in item.items():
                if type(nest_item) is np.float64:
                    new_results[name][nest_name] = float(nest_item)
                elif type(nest_item) is list:
                    new_list = []
                    for val in nest_item:
                        if type(val) is np.int64:
                            new_list.append(int(val))
                        elif type(val) is np.float64:
                            new_list.append(float(val))
                        else:
                            new_list.append(val)
                    new_results[name][nest_name] = new_list
                elif type(nest_item) is np.ndarray:
                    new_list = []
                    for val in nest_item:
                        if type(val) is np.int64:
                            new_list.append(int(val))
                        elif type(val) is np.float64:
                            new_list.append(float(val))
                        else:
                            new_list.append(val)
                    new_results[name][nest_name] = new_list
                elif type(nest_item) is np.int64:
                    new_results[name][nest_name] = int(nest_item)
                else:
                    new_results[name][nest_name] = nest_item
        with open(full_path, 'w') as f:
            json.dump(new_results, f)
        return full_path

    def gen_walk_ds(self, walk_data, ds) -> Dataset:
        dataset = []
        user_data = ds.get_dataset()[0]
        imu_data_file_path: str = user_data.get_imu_data_file_path()
        imu_data_file_name: str = user_data.get_imu_data_file_name()
        imu_metadata_file_path: str = user_data.get_imu_metadata_file_path()
        imu_metadata = user_data.get_imu_metadata()
        trial = ''
        dataset_path = ds.get_dataset_path()
        clinical_demo_path = ds.get_clinical_demo_path()
        clinical_demo_data = user_data.get_clinical_demo_data()
        for walk_bout in walk_data:
            # Build a UserData object for the whole data
            time = np.linspace(0, len(np.array(walk_bout[0])) / int(imu_metadata.get_sampling_frequency()),
                           len(np.array(walk_bout[0])))
            imu_data = self._generate_imu_data_instance(walk_bout, time)
            dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, clinical_demo_path,
                                    {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        return Dataset('LTMM', dataset_path, clinical_demo_path, dataset, {})

    def get_walk_imu_data(self, bout_ixs, ds):
        walk_data = []
        walk_time = []
        imu_data = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.LPF)
        acc_data = imu_data.get_triax_acc_data()
        acc_data = np.array([acc_data['vertical'], acc_data['mediolateral'], acc_data['anteroposterior']])
        for bout_ix in bout_ixs:
            walk_data.append(acc_data[:, bout_ix[0]:bout_ix[1]])
        return walk_data

    def get_walk_bout_ixs(self, skdh_results, ds: Dataset, min_gait_dur):
        gait_results = skdh_results[0]['gait_metrics']
        bout_starts = gait_results['Bout Starts']
        bout_durs = gait_results['Bout Duration']
        t0 = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).get_time()[0]
        bout_ixs = []
        freq = ds.get_dataset()[0].get_imu_metadata().get_sampling_frequency()
        for start, dur in zip(bout_starts, bout_durs):
            if dur > min_gait_dur:
                # Calculate the start and stop ixs of the bout
                st_ix = int((start - t0) * freq)
                end_ix = int(((start + dur) - t0) * freq)
                bout_ixs.append([st_ix, end_ix])
        return bout_ixs

    def preprocess_data(self, dataset):
        freq = dataset.get_dataset()[0].get_imu_metadata().get_sampling_frequency()
        for user_data in dataset.get_dataset():
            # Filter the data
            self.apply_lp_filter(user_data, freq)

    def generate_custom_metrics(self, dataset) -> InputMetrics:
        mg = MetricGenerator()
        return mg.generate_metrics(
            dataset.get_dataset(),
            self.custom_metric_names
        )

    def generate_skdh_metrics(self, dataset, day_ends, pipeline_run: SKDHPipelineRunner, gait=False):
        results = []
        for user_data in dataset.get_dataset():
            # Get the data from the user data in correct format
            # Get the time axis from user data
            # Get sampling rate
            # Generate day ends for the time axes
            imu_data = user_data.get_imu_data(IMUDataFilterType.LPF)
            data = imu_data.get_triax_acc_data()
            data = np.array([data['vertical'], data['mediolateral'], data['anteroposterior']])
            data = data.T
            time = imu_data.get_time()
            fs = user_data.get_imu_metadata().get_sampling_frequency()
            # TODO: create function to translate the time axis into day ends
            # day_ends = np.array([[0, int(len(time) - 1)]])
            if gait:
                results.append(pipeline_run.run_gait_pipeline(data, time, fs))
            else:
                results.append(pipeline_run.run_pipeline(data, time, fs, day_ends))
        return results

    def export_metrics(self, input_metrics: InputMetrics, output_path):
        metric_file_name = 'model_input_metrics_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        full_path = os.path.join(output_path, metric_file_name)
        new_im = {}
        for name, metric in input_metrics.get_metrics().items():
            if isinstance(name, MetricNames):
                new_im[name.value] = metric
            else:
                new_im[name] = metric
        metric_data = {'metrics': [new_im], 'labels': input_metrics.get_labels()}
        with open(full_path, 'w') as f:
            json.dump(metric_data, f)
        return full_path

    def segment_data(self, data, epoch_size, sampling_frequency):
        """
        Segments data into epochs of a given duration starting from the beginning of the data
        :param: data: data to be segmented
        :param epoch_size: duration of epoch to segment data (in seconds)
        :return: data segments of given epoch duration
        """
        total_time = len(data[0])/sampling_frequency
        # Calculate number of segments from given epoch size
        num_of_segs = int(total_time / epoch_size)
        # Check to see if data can be segmented at least one segment of given epoch size
        if num_of_segs > 0:
            data_segments = []
            # Counter for the number of segments to be created
            segment_count = range(0, num_of_segs+1)
            # Create segmentation indices
            seg_ixs = [int(seg * sampling_frequency * epoch_size) for seg in segment_count]
            for seg_num in segment_count:
                if seg_num != segment_count[-1]:
                    data_segments.append(data[:, seg_ixs[seg_num]: seg_ixs[seg_num+1]])
                else:
                    continue
        else:
            raise ValueError(f'Data of total time {str(total_time)}s can not be '
                             f'segmented with given epoch size {str(epoch_size)}s')
        return data_segments

    def apply_lp_filter(self, user_data, freq):
        filter = MotionFilters()
        imu_data: IMUData = user_data.get_imu_data()[IMUDataFilterType.RAW]
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        act_code = imu_data.get_activity_code()
        act_des = imu_data.get_activity_description()
        all_raw_data = imu_data.get_all_data()
        time = imu_data.get_time()
        lpf_data_all_axis = []
        for data in all_raw_data:
            lpf_data = filter.apply_lpass_filter(data, 2, samp_freq) if data.any() else data
            lpf_data_all_axis.append(lpf_data)
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis, time)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data, time):
        activity_code = ''
        activity_description = ''
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array([])
        pitch_gyr_data = np.array([])
        roll_gyr_data = np.array([])
        return IMUData(activity_code, activity_description, v_acc_data,
                       ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                       roll_gyr_data, time)

    def format_input_metrics(self, input_metrics,
                             custom_input_metrics: InputMetrics,
                             skdh_input_metrics):
        for user_metrics in skdh_input_metrics:
            gait_metrics = user_metrics['gait_metrics']
            for name, val in gait_metrics.items():
                if name not in ['Bout Starts', 'Bout Duration']:
                    input_metrics.get_metric(name).append(val)
        for name, metric in custom_input_metrics.get_metrics().items():
            input_metrics.get_metric(name).extend(metric.get_value().tolist())
        final_metrics = InputMetrics()
        for name, metric in input_metrics.get_metrics().items():
            metric = np.array(metric)
            metric = metric[~np.isnan(metric)]
            metric_mean = metric.mean()
            final_metrics.set_metric(name, metric_mean)
        input_metrics.get_labels().extend(custom_input_metrics.get_labels())
        final_metrics.set_labels(input_metrics.get_labels())
        return final_metrics

    def initialize_input_metrics(self, skdh_input_metrics):
        input_metrics = InputMetrics()
        for name in self.custom_metric_names:
            input_metrics.set_metric(name, [])
        for name in skdh_input_metrics[0]['gait_metrics'].keys():
            if name not in ['Bout Starts', 'Bout Duration']:
                input_metrics.set_metric(name, [])
        input_metrics.set_labels([])
        return input_metrics


    def plot_walk_data(self, walk_ds):
        freq = walk_ds.get_dataset()[0].get_imu_metadata().get_sampling_frequency()
        walk_v = []
        walk_ap = []
        walk_ml = []
        for user_data in walk_ds.get_dataset():
            walk_v.extend(user_data.get_imu_data(IMUDataFilterType.LPF).v_acc_data)
            walk_ap.extend(user_data.get_imu_data(IMUDataFilterType.LPF).ap_acc_data)
            walk_ml.extend(user_data.get_imu_data(IMUDataFilterType.LPF).ml_acc_data)
        time = np.linspace(0, len(walk_v) / int(freq),
                           len(walk_v))
        plt.plot(time, walk_v)
        plt.plot(time, walk_ml)
        plt.plot(time, walk_ap)
        plt.show()
        print('Plotting')


def main():
    # Grainger VM paths
    dataset_path = '/home/grainger/Desktop/datasets/fibion/25hz_device/test_Sheedy_2022-08-05.bin'
    mbient_data_path = r'/home/grainger/Desktop/datasets/mbientlab/test/MULTIDAY_MetaWear_2022-08-19T12.38.00.909_C85D72EF7FA2_Accelerometer.csv'
    activity_path = '/home/grainger/Desktop/datasets/fibion/io_test_data/activity/fibion_test_activity_04_10_2022.csv'

    # Grainger desktop paths
    # dataset_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin'
    # activity_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\activity\fibion_test_activity_04_10_2022.csv'

    # Grainger laptop paths
    # dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\bin'
    # activity_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\csv\export_2022-04-11T01_00_00.000000Z.csv'

    # Output path
    output_path = '/home/grainger/Desktop/skdh_testing/'


    demo_data = {'user_height': 1.88}
    # activity_path = r'C:\Users\gsass\Desktop\Fall Project Master\test_data\fibion\csv\2022-04-12_activity_file.csv'
    dataset_name = 'LTMM'
    fib_fafra = FaFRA_SKDH(dataset_path, activity_path, demo_data, output_path)
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

    custom_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/fibion/custom_skdh'
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/fibion/skdh'
    model_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_rcm_20220804-123836.pkl'
    scaler_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_scaler_20220804-123836.bin'
    day_ends = np.array([])
    fib_fafra.perform_risk_assessment(mbient_data_path, demo_data, skdh_path, custom_path, model_path, scaler_path)


if __name__ == '__main__':
    main()

