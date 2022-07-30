import os
import psutil
import numpy as np
import glob
import wfdb
import gc
import time
import json
import joblib
from itertools import repeat
from pympler import asizeof
from typing import List

from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.fibion_mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
# from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier

from src.dataset_tools.dataset_builders.dataset_names import DatasetNames
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_metadata import IMUMetadata
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.clinical_demographic_data import ClinicalDemographicData


class ModelTrainer:
    def __init__(self, dataset_path, clinical_demo_path, segment_dataset,
                 epoch_size, custom_metric_names, gait_metric_names: List[str]):
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.segment_dataset = segment_dataset
        self.epoch_size = epoch_size
        # self.dataset = self.build_dataset()
        self.custom_metric_names = custom_metric_names
        self.gait_metric_names: List[str] = gait_metric_names
        self.rc = LightGBMRiskClassifier({})
        # self.rc = KNNRiskClassifier()
        self.sampling_frequency = 100.0
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        self.height = 1.75

    def generate_model(self, im_path, model_output_path, model_name, scaler_name):
        input_metrics = self.read_parse_im(im_path)
        metrics = input_metrics['metrics'][0]
        labels = input_metrics['labels']
        input_metrics = self.finalize_metric_formatting(metrics, labels)
        eval_results = self.rc.train_model(input_metrics)
        self.test_model(input_metrics)
        # Scale input metrics
        # input_metrics = self.rc.scale_input_data(input_metrics)
        # # Train model
        # self.train_model(input_metrics)
        # # Export model and scaler
        # model_path, scaler_path = self.export_model(model_output_path, model_name, scaler_name)
        # model = self.import_model(model_path)
        # scaler = self.import_model(scaler_path)
        # return None

    def test_model(self, input_metrics):
        x_train, x_test, y_train, y_test = self.rc.split_input_metrics(input_metrics)
        x_train, x_test = self.rc.scale_train_test_data(x_train, x_test)
        self.rc.train_model(x_train, y_train)
        acc = self.rc.score_model(x_test, y_test)
        print('ok')

    def read_parse_im(self, im_path):
        with open(im_path, 'r') as f:
            data = json.load(f)
        return data

    def finalize_metric_formatting(self, metrics, labels):
        new_ims = InputMetrics()
        for name, metric in metrics.items():
            new_metric = []
            new_metric.extend(repeat(metric, 5))
            new_metric = [item for sublist in new_metric for item in sublist]
            im = InputMetric(name, np.array(metric))
            # im = InputMetric(name, np.array(new_metric))
            new_ims.set_metric(name, im)
        new_labels = []
        new_labels.extend(repeat(labels, 5))
        new_labels = [item for sublist in new_labels for item in sublist]
        new_ims.set_labels(np.array(labels))
        # new_ims.set_labels(np.array(new_labels))
        return new_ims

    def train_model(self, input_metrics):
        x, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        self.rc.train_model(x, y, metric_names=names)

    def export_model(self, model_output_path, model_name, scaler_name):
        model_name = model_name + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
        scaler_name = scaler_name + time.strftime("%Y%m%d-%H%M%S") + '.bin'
        model_path = os.path.join(model_output_path, model_name)
        scaler_path = os.path.join(model_output_path, scaler_name)
        # self.rc.model.save_model(model_path)
        joblib.dump(self.rc.get_model(), model_path)
        joblib.dump(self.rc.get_scaler(), scaler_path)
        return model_path, scaler_path

    def import_model(self, model_path):
        model = joblib.load(model_path)
        return model

    def generate_input_metrics(self, skdh_output_path, im_path):
        time0 = time.time()
        pid = os.getpid()
        ps = psutil.Process(pid)
        head_df_paths = self._generate_header_and_data_file_paths()
        input_metrics = self.initialize_input_metrics()
        pipeline_gen = SKDHPipelineGenerator()
        full_pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        full_pipeline_run = SKDHPipelineRunner(full_pipeline)
        gait_pipeline = pipeline_gen.generate_gait_pipeline(skdh_output_path)
        gait_pipeline_run = SKDHPipelineRunner(gait_pipeline)
        for name, header_and_data_file_path in head_df_paths.items():
            # Load the data and compute the input metrics for the file
            ds = self.create_dataset(header_and_data_file_path)
            print(str(asizeof.asizeof(ds) / 100000000))
            self.preprocess_data(ds)
            print(str(asizeof.asizeof(ds) / 100000000))
            custom_input_metrics: InputMetrics = self.generate_custom_metrics(ds)
            skdh_input_metrics = self.generate_skdh_metrics(ds, gait_pipeline_run, True)
            bout_ixs = self.get_walk_bout_ixs(skdh_input_metrics, ds)
            walk_data = self.get_walk_imu_data(bout_ixs, ds)
            input_metrics = self.format_input_metrics(input_metrics,
                                                      custom_input_metrics, skdh_input_metrics)
            del ds
            gc.collect()
            memory_usage = ps.memory_info()
            print('\n')
            print('\n')
            print(memory_usage)
            print('\n')
            print('\n')
        full_path = self.export_metrics(input_metrics, im_path)
        print(time.time() - time0)
        print(input_metrics.get_metrics())
        return full_path

    def get_walk_imu_data(self, bout_ixs, ds):
        walk_data = []
        walk_time = []
        imu_data = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.LPF)
        acc_data = imu_data.get_triax_acc_data()
        acc_data = np.array([acc_data['vertical'], acc_data['mediolateral'], acc_data['anteroposterior']])
        for bout_ix in bout_ixs:
            walk_data.append(acc_data[:, bout_ix[0]:bout_ix[1]])
        return walk_data

    def get_walk_bout_ixs(self, skdh_results, ds):
        bout_starts = skdh_results['Bout Starts']
        bout_durs = skdh_results['Bout Duration']
        t0 = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.RAW).get_time()[0]
        bout_ixs = []
        for start, dur in zip(bout_starts, bout_durs):
            # Calculate the start and stop ixs of the bout
            st_ix = int((start - t0) * self.sampling_frequency)
            end_ix = int(((start + dur) - t0) * self.sampling_frequency)
            bout_ixs.append([st_ix, end_ix])
        return bout_ixs

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
            # Adding time to make this a realistic epoch
            time = time + 1658333118.0
            fs = user_data.get_imu_metadata().get_sampling_frequency()
            # TODO: create function to translate the time axis into day ends
            day_ends = np.array([[0, int(len(time) - 1)]])
            if gait:
                results.append(pipeline_run.run_gait_pipeline(data, time, fs, day_ends))
            else:
                results.append(pipeline_run.run_pipeline(data, time, fs))
        return results

    def preprocess_data(self, dataset):
        for user_data in dataset.get_dataset():
            # Filter the data
            self.apply_lp_filter(user_data)

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
        return IMUData(activity_code, activity_description, v_acc_data,
                       ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                       roll_gyr_data, time)

    def initialize_input_metrics(self):
        input_metrics = InputMetrics()
        for name in self.custom_metric_names:
            input_metrics.set_metric(name, [])
        for name in self.gait_metric_names:
            input_metrics.set_metric(name, [])
        input_metrics.set_labels([])
        return input_metrics

    def format_input_metrics(self, input_metrics,
                             custom_input_metrics: InputMetrics,
                             skdh_input_metrics):
        for user_metrics in skdh_input_metrics:
            gait_metrics = user_metrics['gait_metrics']
            for name, val in gait_metrics.items():
                if name in self.gait_metric_names:
                    input_metrics.get_metric(name).append(val)
        for name, metric in custom_input_metrics.get_metrics().items():
            input_metrics.get_metric(name).extend(metric.get_value().tolist())
        input_metrics.get_labels().extend(custom_input_metrics.get_labels())
        return input_metrics

    # def build_dataset(self):
    #     db = DatasetBuilder()
    #     return db.build_dataset(
    #         self.dataset_path,
    #         self.clinical_demo_path,
    #         self.segment_dataset,
    #         self.epoch_size
    #     )
    #
    # def format_input_metrics(self, custom_input_metrics: InputMetrics,
    #                          skdh_input_metrics):
    #     custom_metrics = custom_input_metrics.get_metric_matrix()
    #     custom_metrics = np.reshape(custom_metrics, (1, -1))
    #     model_gait_metrics = {
    #         'PARAM:gait speed: mean': [],
    #         'PARAM:gait speed: std': [],
    #         'BOUTPARAM:gait symmetry index: mean': [],
    #         'BOUTPARAM:gait symmetry index: std': [],
    #         'PARAM:cadence: mean': [],
    #         'PARAM:cadence: std': []
    #     }
    #     for user_metrics in skdh_input_metrics:
    #         gait_metrics = user_metrics['gait_metrics']
    #         for name, val in gait_metrics.items():
    #             if name in model_gait_metrics.keys():
    #                 model_gait_metrics[name].append(val)
    #     for name, val in model_gait_metrics.items():
    #         im = InputMetric(name, np.array(val))
    #         custom_input_metrics.set_metric(name, im)
    #     print(model_gait_metrics)

        # Scale the input metrics
    #
    # def train_model(self, input_metrics, input_metric_names):
    #     x, names = input_metrics.get_metric_matrix()
    #     y = input_metrics.get_labels()
    #     self.rc.train_model(x, y, metric_names=input_metric_names)

    # def export_model(self, model_name, scaler_name):
    #     model_path = os.path.join(self.output_path, model_name)
    #     scaler_path = os.path.join(self.output_path, scaler_name)
    #     # self.rc.model.save_model(model_path)
    #     joblib.dump(self.rc.get_model(), model_path)
    #     joblib.dump(self.rc.get_scaler(), scaler_path)
    #     return model_path
    #
    # def import_model(self, model_path):
    #     model = joblib.load(model_path)
    #     return model


def main():
    dp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/'
    cdp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    metric_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/'
    seg = False
    epoch = 0.0
    # metric_names = tuple(
    #     [
    #         MetricNames.AUTOCORRELATION,
    #         MetricNames.FAST_FOURIER_TRANSFORM,
    #         MetricNames.MEAN,
    #         MetricNames.ROOT_MEAN_SQUARE,
    #         MetricNames.STANDARD_DEVIATION,
    #         MetricNames.SIGNAL_ENERGY,
    #         MetricNames.COEFFICIENT_OF_VARIANCE,
    #         MetricNames.ZERO_CROSSING,
    #         MetricNames.SIGNAL_MAGNITUDE_AREA
    #     ]
    # )
    custom_metric_names = tuple(
        [
            MetricNames.SIGNAL_MAGNITUDE_AREA,
            MetricNames.COEFFICIENT_OF_VARIANCE,
            MetricNames.STANDARD_DEVIATION,
            MetricNames.MEAN,
            MetricNames.SIGNAL_ENERGY,
            MetricNames.ROOT_MEAN_SQUARE
        ]
    )
    gait_metric_names = [
            'PARAM:gait speed: mean',
            'PARAM:gait speed: std',
            'BOUTPARAM:gait symmetry index: mean',
            'BOUTPARAM:gait symmetry index: std',
            'PARAM:cadence: mean',
            'PARAM:cadence: std'
        ]
    mt = ModelTrainer(dp, cdp, seg, epoch, custom_metric_names, gait_metric_names)
    full_path = mt.generate_input_metrics(
        '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/',
        '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/custom_skdh/'
    )
    # im_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/custom_skdh/model_input_metrics_20220726-152733.json'
    # model_output_path = ''
    file_name = ''
    # model_name = 'lgbm_skdh_ltmm_rcm_'
    # scaler_name = 'lgbm_skdh_ltmm_scaler_'
    # mt.generate_model(im_path, model_output_path, model_name, scaler_name)


if __name__ == '__main__':
    main()

