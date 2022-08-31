import os
import numpy as np
import json
import time
import joblib
from typing import List


from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.mvp.report_generation.report_generator import ReportGenerator
from src.mvp.mbientlab_dataset_builder import MbientlabDatasetBuilder
from src.mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner


class FaFRA:
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
            # Additional gait params
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

    def perform_risk_assessment(self, assessment_path, ra_model_path, ra_scaler_path):
        # Generate risk metrics
        ra_metrics_path, ra_metrics = MetricGen().generate_ra_metrics(
            assessment_path, self.custom_metric_names, self.gait_metric_names)
        # Assess risk using risk model
        ra_results = Model().assess_fall_risk(ra_model_path, ra_scaler_path, ra_metrics)
        # Generate risk report
        rg = ReportGenerator()
        rg.generate_report(assessment_path, '', '', '')


class MetricGen:
    def generate_ra_metrics(self, assessment_path, custom_metric_names, gait_metric_names):
        ds = DataLoader().load_data(assessment_path)
        # Preprocess data
        self.preprocess_data(ds)
        # TODO: Get day_ends from data
        day_ends = np.array([[0, 3836477], [3836477, 7607840]])
        # Segment data along walking bouts
        walk_ds = self.segment_data_walk(ds, gait_metric_names, day_ends, assessment_path)
        # Generate metrics on walking data
        custom_input_metrics: InputMetrics = self.generate_custom_metrics(walk_ds, custom_metric_names)
        pipeline_gen = SKDHPipelineGenerator()
        gait_pipeline = pipeline_gen.generate_gait_pipeline()
        gait_pipeline_run = SKDHPipelineRunner(gait_pipeline, gait_metric_names)
        skdh_input_metrics = self.generate_skdh_metrics(walk_ds, day_ends, gait_pipeline_run, True)
        # Format input metrics
        input_metrics = self.format_input_metrics(custom_input_metrics, skdh_input_metrics, custom_metric_names)
        # Export metrics
        full_path = self.export_metrics(input_metrics, assessment_path)
        return full_path, input_metrics

    def preprocess_data(self, dataset):
        freq = dataset.get_dataset()[0].get_imu_metadata().get_sampling_frequency()
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

    def segment_data_walk(self, ds, gait_metric_names, day_ends, assessment_path):
        skdh_output_path = ''
        # Run initial pass of SKDH on data
        pipeline_gen = SKDHPipelineGenerator()
        # TODO: Set output path
        full_pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        full_pipeline_run = SKDHPipelineRunner(full_pipeline, gait_metric_names)
        skdh_input_metrics = self.generate_skdh_metrics(ds, day_ends, full_pipeline_run, False)
        self.export_skdh_results(skdh_input_metrics, skdh_output_path)
        bout_ixs = self.get_walk_bout_ixs(skdh_input_metrics, ds, 30.0)
        if bout_ixs:
            walk_data = self.get_walk_imu_data(bout_ixs, ds)
            # Create new dataset from the walking data segments
            walk_ds = self.gen_walk_ds(walk_data, ds)
            self.preprocess_data(walk_ds)
        else:
            print('FAILED TO SEGMENT DATA ALONG GAIT BOUTS')
        return walk_ds

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
        ds_name = ds.get_dataset_name()
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
        return Dataset(ds_name, dataset_path, clinical_demo_path, dataset, {})

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

    def get_walk_imu_data(self, bout_ixs, ds):
        walk_data = []
        walk_time = []
        imu_data = ds.get_dataset()[0].get_imu_data(IMUDataFilterType.LPF)
        acc_data = imu_data.get_triax_acc_data()
        acc_data = np.array([acc_data['vertical'], acc_data['mediolateral'], acc_data['anteroposterior']])
        for bout_ix in bout_ixs:
            walk_data.append(acc_data[:, bout_ix[0]:bout_ix[1]])
        return walk_data

    def generate_custom_metrics(self, dataset, custom_metric_names) -> InputMetrics:
        mg = MetricGenerator()
        return mg.generate_metrics(
            dataset.get_dataset(),
            custom_metric_names
        )

    def format_input_metrics(self, custom_input_metrics: InputMetrics,
                             skdh_input_metrics, custom_metric_names):
        input_metrics = self.initialize_input_metrics(skdh_input_metrics, custom_metric_names)
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

    def initialize_input_metrics(self, skdh_input_metrics, custom_metric_names):
        input_metrics = InputMetrics()
        for name in custom_metric_names:
            input_metrics.set_metric(name, [])
        for name in skdh_input_metrics[0]['gait_metrics'].keys():
            if name not in ['Bout Starts', 'Bout Duration']:
                input_metrics.set_metric(name, [])
        input_metrics.set_labels([])
        return input_metrics

    def export_metrics(self, input_metrics: InputMetrics, assessment_path):
        output_path = os.path.join(assessment_path, 'generated_data', 'ra_model_metrics')
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


class DataLoader:
    def load_json_data(self, file_path):
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_data(self, assessment_path):
        # Load in the assessment data file (contains device type of the IMU data)
        assess_info_path = os.path.join(assessment_path, 'assessment_info.json')
        assess_info = self.load_json_data(assess_info_path)
        data_type = assess_info['device_type']
        # Load in the user data json file (contains demographic data of the user)
        collect_data_path = os.path.join(assessment_path, 'collected_data')
        user_info_path = os.path.join(collect_data_path, 'user_info.json')
        user_info = self.load_json_data(user_info_path)
        # Load in the IMU data file based on type
        imu_data_filename = [filename for filename in os.listdir(collect_data_path) if filename.startswith("imu_data")][0]
        imu_data_path = os.path.join(collect_data_path, imu_data_filename)
        # Build dataset objects
        return self.build_dataset('mbientlab', imu_data_path, user_info)

    def build_dataset(self, data_type, imu_data_path, user_info):
        if data_type.lower() == 'mbientlab':
            user_data: List[UserData] = MbientlabDatasetBuilder().build_single_user(imu_data_path, user_info)
            # TODO: may need to define ra data objects specific to the MVP
        else:
            raise ValueError(f'Unknown IMU datatype provided {data_type}')
        dataset = Dataset(
            'mbientlab', [imu_data_path], [], user_data, {}
        )
        return dataset


class Model:
    def assess_fall_risk(self, model_path, scaler_path, metrics):
        risk_model = self.import_classifier(model_path, scaler_path)
        metrics = self.format_input_metrics_scaling(metrics)
        metrics = risk_model.scaler.transform(metrics)
        prediction = risk_model.make_prediction(metrics)[0]
        return prediction

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


def main():
    fafra = FaFRA()
    assessment_path = '/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/'
    ra_model_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_rcm_20220804-123836.pkl'
    ra_scaler_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_scaler_20220804-123836.bin'
    ra = fafra.perform_risk_assessment(assessment_path, ra_model_path, ra_scaler_path)


if __name__ == '__main__':
    main()
