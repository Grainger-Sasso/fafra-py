import os
import time
import numpy as np
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

    def perform_risk_assessment(self, data_path, demographic_data, output_path, model_path, scaler_path):
        # Load in the accelerometer data
        user_data = self.load_dataset(data_path, demographic_data)
        # Generate SKDH metrics on the user data
        # Assess risk levels using risk model
        pass

    def load_dataset(self, dataset_path, demo_data):
        fdb = FibionDatasetBuilder()
        ds = fdb.build_single_user(dataset_path, demo_data)
        print('done')
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

    def generate_pipeline(self):
        pipeline = Pipeline()
        # gait_file = os.path.join(self.output_path, 'gait_results.csv')
        # pipeline.add(Gait(), save_file=gait_file)
        act_file = os.path.join(self.output_path, 'activity_results.csv')
        pipeline.add(ActivityLevelClassification(), save_file=act_file)
        # sleep_file = os.path.join(self.output_path, 'sleep_results.csv')
        # pipeline.add(Sleep(), save_file=sleep_file)
        return pipeline

    def perform_risk_analysis(self, input_metric_names=tuple(MetricNames.get_all_enum_entries())):
        t0 = time.time()
        # Preprocess subject data (low-pass filtering)
        self.preprocess_data()
        # Generate SKDH pipeline
        pipeline = self.generate_pipeline()
        # Run SKDH Pipeline
        self.run_pipeline(pipeline)
        # Estimate subject fall risk
        fall_risk_score = self.estimate_fall_risk(input_metric_names, gait_speed)
        # Evaluate user fall risk status
        # Evaluate faller risk levels
        # Build risk report
        self.build_risk_report(fall_risk_score)

    def run_pipeline(self, pipeline: Pipeline):
        # Get numpy 3D array of IMU data for each epoch in dataset
        user_data = self.dataset.get_dataset()[0]
        imu_data = user_data.get_imu_data(IMUDataFilterType.LPF)
        tri_acc = imu_data.get_triax_acc_data()
        tri_acc = np.array([tri_acc['vertical'], tri_acc['mediolateral'], tri_acc['anteroposterior']])
        time = imu_data.get_time()
        pipeline.run(time=time, accel=tri_acc)
        print('yahoo')

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
        time = imu_data.get_time()
        lpf_data_all_axis = []
        for data in all_raw_data:
            lpf_data = self.filter.apply_lpass_filter(data, 2, samp_freq) if data.any() else data
            lpf_data_all_axis.append(lpf_data)
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis,
                                                        samp_freq, act_code,
                                                        act_des, time)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data, sampling_freq, act_code, act_des, time):
        # TODO: Finish reformatting the imu data for new data instances after lpf
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
        # time = np.linspace(0, len(v_acc_data) / int(sampling_freq),
        #                    len(v_acc_data))
        return IMUData(act_code, act_des, v_acc_data, ml_acc_data, ap_acc_data,
                       yaw_gyr_data, pitch_gyr_data, roll_gyr_data, time)


class FibionMetricGenerator:
    def __init__(self, dataset_path, clinical_demo_path, segment_dataset,
                 epoch_size, custom_metric_names, gait_metric_names: List[str], final_skdh_metric_names):
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.segment_dataset = segment_dataset
        self.epoch_size = epoch_size
        # self.dataset = self.build_dataset()
        self.custom_metric_names = custom_metric_names
        self.gait_metric_names: List[str] = gait_metric_names
        self.head_df_paths = self._generate_header_and_data_file_paths()
        self.running_analysis_total = 0
        self.bout_segmented_total = 0
        self.bout_seg_fail_total = 0
        self.sampling_frequency = 100.0
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        self.height = 1.75

    def generate_input_metrics(self, skdh_output_path, im_path, seg_gait=True, min_gait_dur=30.0):
        time0 = time.time()
        pid = os.getpid()
        ps = psutil.Process(pid)
        pipeline_gen = SKDHPipelineGenerator()
        full_pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        full_pipeline_run = SKDHPipelineRunner(full_pipeline, self.gait_metric_names)
        gait_pipeline = pipeline_gen.generate_gait_pipeline(skdh_output_path)
        gait_pipeline_run = SKDHPipelineRunner(gait_pipeline, self.gait_metric_names)
        input_metrics = None
        for name, header_and_data_file_path in self.head_df_paths.items():
            self.running_analysis_total += 1
            walk_ds = None
            # Load the data and compute the input metrics for the file
            ds = self.create_dataset(header_and_data_file_path)
            print(f'Dataset size: {str(asizeof.asizeof(ds) / 100000000)}')
            self.preprocess_data(ds)
            print(f'Dataset size: {str(asizeof.asizeof(ds) / 100000000)}')
            custom_input_metrics: InputMetrics = self.generate_custom_metrics(ds)
            skdh_input_metrics = self.generate_skdh_metrics(ds, full_pipeline_run, False)
            if not input_metrics:
                input_metrics = self.initialize_input_metrics(skdh_input_metrics)
            # If data is to be segmented along gait data, regenerate dataset using walking data and
            if seg_gait:
                bout_ixs = self.get_walk_bout_ixs(skdh_input_metrics, ds, min_gait_dur)
                if bout_ixs:
                    self.bout_segmented_total += 1
                    print(f'Percentage of data segmented by walking data: {(self.bout_segmented_total/self.running_analysis_total) * 100.0}')
                    walk_data = self.get_walk_imu_data(bout_ixs, ds)
                    # Create new dataset from the walking data segments
                    walk_ds = self.gen_walk_ds(walk_data, ds)
                    self.preprocess_data(walk_ds)
                    walk_ds_len = 0
                    for user_data in walk_ds.get_dataset():
                        walk_ds_len += len(user_data.get_imu_data(IMUDataFilterType.RAW).v_acc_data)
                    ds_len = len(ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).v_acc_data)
                    print(f'Percentage of walking-segmented data to recorded data: {(walk_ds_len/ds_len) * 100.0}')
                    custom_input_metrics: InputMetrics = self.generate_custom_metrics(walk_ds)
                    skdh_input_metrics = self.generate_skdh_metrics(walk_ds, gait_pipeline_run, True)
                else:
                    print(f'Unable to segment file for {min_gait_dur}, percentage of failed segmentations: {(self.bout_seg_fail_total/self.running_analysis_total) * 100.0}')
            input_metrics = self.format_input_metrics(input_metrics,
                                                      custom_input_metrics, skdh_input_metrics)
            del ds
            if walk_ds:
                del walk_ds
            gc.collect()
            memory_usage = ps.memory_info()
            print(memory_usage)
            print('\n')
            print('\n')
        full_path = self.export_metrics(input_metrics, im_path)
        print(time.time() - time0)
        print(
            f'Percentage of data segmented by walking data: {(self.bout_segmented_total / self.running_analysis_total) * 100.0}')
        print(
            f'Percentage of failed segmentations: {(self.bout_seg_fail_total / self.running_analysis_total) * 100.0}')
        print(input_metrics.get_metrics())
        return full_path

    def gen_walk_ds(self, walk_data, ds) -> Dataset:
        dataset = []
        user_data = ds.get_dataset()[0]
        imu_data_file_path: str = user_data.get_imu_data_file_path()
        imu_data_file_name: str = user_data.get_imu_data_file_name()
        imu_metadata_file_path: str = user_data.get_imu_metadata_file_path()
        imu_metadata = user_data.get_imu_metadata()
        trial = ''
        clinical_demo_data = user_data.get_clinical_demo_data()
        for walk_bout in walk_data:
            # Build a UserData object for the whole data
            imu_data = self._generate_imu_data_instance(walk_bout)
            dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, self.clinical_demo_path,
                                    {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        return Dataset('LTMM', self.dataset_path, self.clinical_demo_path, dataset, {})

    def get_walk_imu_data(self, bout_ixs, ds):
        walk_data = []
        walk_time = []
        imu_data = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.LPF)
        acc_data = imu_data.get_triax_acc_data()
        acc_data = np.array([acc_data['vertical'], acc_data['mediolateral'], acc_data['anteroposterior']])
        for bout_ix in bout_ixs:
            walk_data.append(acc_data[:, bout_ix[0]:bout_ix[1]])
        return walk_data

    def get_walk_bout_ixs(self, skdh_results, ds, min_gait_dur):
        gait_results = skdh_results[0]['gait_metrics']
        bout_starts = gait_results['Bout Starts']
        bout_durs = gait_results['Bout Duration']
        t0 = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).get_time()[0]
        bout_ixs = []
        for start, dur in zip(bout_starts, bout_durs):
            if dur > min_gait_dur:
                # Calculate the start and stop ixs of the bout
                st_ix = int((start - t0) * self.sampling_frequency)
                end_ix = int(((start + dur) - t0) * self.sampling_frequency)
                bout_ixs.append([st_ix, end_ix])
        return bout_ixs

    def _generate_header_and_data_file_paths(self):
        header_and_data_file_paths = dict()
        data_file_paths = {}
        header_file_paths = {}
        # Get all data file paths
        for data_file_path in glob.glob(os.path.join(self.dataset_path, '*.dat')):
            data_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
            data_file_paths[data_file_name] = data_file_path
        # Get all header file paths
        for header_file_path in glob.glob(os.path.join(self.dataset_path, '*.hea')):
            header_file_name = os.path.splitext(os.path.basename(header_file_path))[0]
            header_file_paths[header_file_name] = header_file_path
        # Match corresponding data and header files
        for name, path in data_file_paths.items():
            corresponding_header_file_path = header_file_paths[name]
            header_and_data_file_paths[name] = {'data_file_path': path,
                                                     'header_file_path': corresponding_header_file_path}
        return header_and_data_file_paths

    def initialize_input_metrics(self, skdh_input_metrics):
        input_metrics = InputMetrics()
        for name in self.custom_metric_names:
            input_metrics.set_metric(name, [])
        for name in skdh_input_metrics[0]['gait_metrics'].keys():
            if name not in ['Bout Starts', 'Bout Duration']:
                input_metrics.set_metric(name, [])
        input_metrics.set_labels([])
        return input_metrics

    def create_dataset(self, header_and_data_file_path):
        dataset = []
        data_file_path = header_and_data_file_path['data_file_path']
        header_file_path = header_and_data_file_path['header_file_path']
        data_path = os.path.splitext(data_file_path)[0]
        header_path = os.path.splitext(header_file_path)[0]
        wfdb_record = wfdb.rdrecord(data_path)
        id = wfdb_record.record_name
        print(id)
        data = np.array(wfdb_record.p_signal, dtype=np.float16)
        data = np.float16(data)
        # DO NOT CONVERT FOR SKDH PIPELINE DATA MAKE SURE DATA IS IN 'g'
        # Convert acceleration data from g to m/s^2
        # data[:, 0:3] = data[:, 0:3] * 9.80665
        header_data = wfdb.rdheader(header_path)
        if wfdb_record.comments[0][4:]:
            age = float(wfdb_record.comments[0][4:])
        else:
            age = 70.0
        sex = wfdb_record.comments[1][4:]
        if id.casefold()[0] == 'f':
            faller_status = True
        elif id.casefold()[0] == 'c':
            faller_status = False
        else:
            raise ValueError('LTMM Data faller status unclear from id')
        imu_data_file_path: str = data_file_path
        imu_data_file_name: str = os.path.split(os.path.splitext(imu_data_file_path)[0])[1]
        imu_metadata_file_path: str = header_file_path
        imu_metadata = IMUMetadata(header_data, self.sampling_frequency, self.units)
        trial = ''
        clinical_demo_data = ClinicalDemographicData(id, age, sex, faller_status, self.height, trial)
        if self.segment_dataset:
            # TODO: fix dataset segmentation
            # Segment the data and build a UserData object for each epoch
            data_segments = self.segment_data(data.T, self.epoch_size, self.sampling_frequency)
            for segment in data_segments:
                imu_data = self._generate_imu_data_instance(segment)
                dataset.append(
                    UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, self.clinical_demo_path,
                             {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        else:
            # Build a UserData object for the whole data
            imu_data = self._generate_imu_data_instance(data.T)
            dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, self.clinical_demo_path,
                                    {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        del data
        return Dataset('LTMM', self.dataset_path, self.clinical_demo_path, dataset, {})

    def preprocess_data(self, dataset):
        for user_data in dataset.get_dataset():
            # Filter the data
            self.apply_lp_filter(user_data)

    def generate_custom_metrics(self, dataset) -> InputMetrics:
        mg = MetricGenerator()
        return mg.generate_metrics(
            dataset.get_dataset(),
            self.custom_metric_names
        )

    def generate_skdh_metrics(self, dataset, pipeline_run: SKDHPipelineRunner, gait=False):
        results = []
        for user_data in dataset.get_dataset():
            # Get the data from the user data in correct format
            # Get the time axis from user data
            # Get sampling rate
            # Generate day ends for the time axes
            imu_data = user_data.get_imu_data(IMUDataFilterType.RAW)
            data = imu_data.get_triax_acc_data()
            data = np.array([data['vertical'], data['mediolateral'], data['anteroposterior']])
            data = data.T
            time = imu_data.get_time()
            fs = user_data.get_imu_metadata().get_sampling_frequency()
            # TODO: create function to translate the time axis into day ends
            day_ends = np.array([[0, int(len(time) - 1)]])
            if gait:
                results.append(pipeline_run.run_gait_pipeline(data, time, fs, day_ends))
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

    def apply_lp_filter(self, user_data):
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
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

    def _generate_imu_data_instance(self, data):
        activity_code = ''
        activity_description = ''
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = []
        pitch_gyr_data = []
        roll_gyr_data = []
        time = np.linspace(0, len(v_acc_data) / int(self.sampling_frequency),
                           len(v_acc_data))
        # Adding time to make this a realistic epoch
        time = time + 1658333118.0
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
        input_metrics.get_labels().extend(custom_input_metrics.get_labels())
        return input_metrics

    def plot_walk_data(self, walk_ds):
        walk_v = []
        walk_ap = []
        walk_ml = []
        for user_data in walk_ds.get_dataset():
            walk_v.extend(user_data.get_imu_data(IMUDataFilterType.LPF).v_acc_data)
            walk_ap.extend(user_data.get_imu_data(IMUDataFilterType.LPF).ap_acc_data)
            walk_ml.extend(user_data.get_imu_data(IMUDataFilterType.LPF).ml_acc_data)
        time = np.linspace(0, len(walk_v) / int(self.sampling_frequency),
                           len(walk_v))
        plt.plot(time, walk_v)
        plt.plot(time, walk_ml)
        plt.plot(time, walk_ap)
        plt.show()
        print('Plotting')


def main():
    # Grainger VM paths
    dataset_path = '/home/grainger/Desktop/datasets/fibion/25hz_device/test_Sheedy_2022-08-05.bin'
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

    fib_fafra.perform_risk_assessment(dataset_path, demo_data ,'', '', '')


if __name__ == '__main__':
    main()

