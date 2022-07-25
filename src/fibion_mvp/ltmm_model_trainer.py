import os
import psutil
import numpy as np
import glob
import wfdb
import gc
import time
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
        self.sampling_frequency = 100.0
        self.units = {'vertical-acc': 'm/s^2', 'mediolateral-acc': 'm/s^2',
                      'anteroposterior-acc': 'm/s^2',
                      'yaw': '°/s', 'pitch': '°/s', 'roll': '°/s'}
        self.height = 1.75

    def generate_model(self, skdh_output_path, model_output_path, file_name):
        time0 = time.time()
        input_metrics = self.generate_input_metrics(skdh_output_path)
        print(time.time() - time0)
        print(input_metrics.get_metrics())
        print('HUH?')
        # Scale input metrics
        # Preprocess data
        # self.preprocess_data()
        # # Generate custom metrics
        # custom_input_metrics: InputMetrics = self.generate_custom_metrics()
        # # print(custom_input_metrics)
        # # Generate SKDH metrics
        # skdh_input_metrics = self.generate_skdh_metrics(skdh_output_path)
        # # print(skdh_input_metrics)
        # # Format input metrics
        # input_metrics = self.format_input_metrics(custom_input_metrics, skdh_input_metrics)
        # # Train model on input metrics
        # # Export model, scaler
        # # Export traning data
        # pass

    def generate_input_metrics(self, skdh_output_path):
        pid = os.getpid()
        ps = psutil.Process(pid)
        head_df_paths = self._generate_header_and_data_file_paths()
        input_metrics = self.initialize_input_metrics()
        pipeline_gen = SKDHPipelineGenerator()
        pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        pipeline_run = SKDHPipelineRunner(pipeline)
        for name, header_and_data_file_path in head_df_paths.items():
            # Load the data and compute the input metrics for the file
            ds = self.create_dataset(header_and_data_file_path)
            print(str(int(asizeof.asizeof(ds) / 100000000)))
            self.preprocess_data(ds)
            print(str(int(asizeof.asizeof(ds) / 100000000)))
            custom_input_metrics: InputMetrics = self.generate_custom_metrics(ds)
            skdh_input_metrics = self.generate_skdh_metrics(ds, pipeline_run)
            input_metrics = self.format_input_metrics(input_metrics,
                                                      custom_input_metrics, skdh_input_metrics)
            del ds
            gc.collect()
            memory_usage = ps.memory_info()
            print('\n')
            print('\n')
            print(memory_usage)
            print(input_metrics.get_metrics())
            print('\n')
            print('\n')
        input_metrics = self.finalize_metric_formatting(input_metrics)
        return self.rc.scale_input_data(input_metrics)

    def create_dataset(self, header_and_data_file_path):
        dataset = []
        data_file_path = header_and_data_file_path['data_file_path']
        header_file_path = header_and_data_file_path['header_file_path']
        data_path = os.path.splitext(data_file_path)[0]
        header_path = os.path.splitext(header_file_path)[0]
        wfdb_record = wfdb.rdrecord(data_path)
        id = wfdb_record.record_name
        print(id)
        data = np.array(wfdb_record.p_signal)
        data = np.float16(data)
        # Convert acceleration data from g to m/s^2
        data[:, 0:3] = data[:, 0:3] * 9.80665
        header_data = wfdb.rdheader(header_path)
        if wfdb_record.comments[0][4:]:
            age = float(wfdb_record.comments[0][4:])
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
        imu_data = self._generate_imu_data_instance(data.T)
        dataset.append(UserData(imu_data_file_path, imu_data_file_name, imu_metadata_file_path, self.clinical_demo_path,
                                {IMUDataFilterType.RAW: imu_data}, imu_metadata, clinical_demo_data))
        return Dataset('LTMM', self.dataset_path, self.clinical_demo_path, dataset, {})

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

    def generate_skdh_metrics(self, dataset, pipeline_run: SKDHPipelineRunner):
        results = []
        for user_data in dataset.get_dataset():
            print(user_data.get_clinical_demo_data().get_id())
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
            results.append(pipeline_run.run_pipeline(data, time, fs, day_ends))
            # results.append(pipeline_run.run_pipeline(data, time, fs))
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
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
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

    def finalize_metric_formatting(self, input_metrics: InputMetrics):
        new_ims = InputMetrics
        for name, metric in input_metrics.get_metrics().items():
            im = InputMetric(name, np.array(metric))
            new_ims.set_metric(name, im)
        new_ims.set_labels(np.array(input_metrics.get_labels()))
        return new_ims

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
            MetricNames.ZERO_CROSSING,
            MetricNames.SIGNAL_ENERGY,
            MetricNames.ROOT_MEAN_SQUARE,
            MetricNames.FAST_FOURIER_TRANSFORM
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
    mt.generate_model('','','')
    print('yup')


if __name__ == '__main__':
    main()

