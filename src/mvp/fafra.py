import os
import numpy as np
import json
import time
import joblib
from datetime import datetime
from typing import List


from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.mvp.report_generation.report_generator import ReportGenerator
from src.mvp.mbientlab_dataset_builder import MbientlabDatasetBuilder
from src.mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner
from src.mvp.fafra_path_handler import PathHandler


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
        path_handler = PathHandler(assessment_path)
        # Generate risk metrics
        ra_metrics_path, ra_metrics = MetricGen().generate_ra_metrics(
            path_handler, self.custom_metric_names, self.gait_metric_names)
        # Assess risk using risk model
        ra_results_file, ra_results = Model().assess_fall_risk(ra_model_path, ra_scaler_path, ra_metrics, path_handler)
        # Generate risk report
        rg = ReportGenerator()
        rg.generate_report(path_handler, ra_results)


class MetricGen:
    def generate_ra_metrics(self, path_handler: PathHandler, custom_metric_names, gait_metric_names):
        assessment_path = path_handler.assessment_folder
        ds = DataLoader().load_data(path_handler)
        # Preprocess data
        self.preprocess_data(ds)
        day_ends = self.get_day_ends(ds)
        # Segment dataset on walking bouts, return new dataset and SKDH results
        walk_ds, skdh_input_metrics = self.segment_data_walk(ds, gait_metric_names, day_ends, path_handler)
        # Generate metrics on walking data
        custom_input_metrics: InputMetrics = self.generate_custom_metrics(walk_ds, custom_metric_names)
        # Format input metrics
        ra_metrics = self.format_input_metrics(custom_input_metrics, skdh_input_metrics, custom_metric_names)
        # Export metrics
        ra_metrics_path = self.export_metrics(ra_metrics, path_handler)
        path_handler.ra_metrics_file = ra_metrics_path
        return ra_metrics_path, ra_metrics

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
        return np.array(day_end_pairs)

    def preprocess_data(self, dataset):
        for user_data in dataset.get_dataset():
            # Filter the data
            self.apply_lp_filter(user_data)

    def apply_lp_filter(self, user_data):
        filter = MotionFilters()
        imu_data: IMUData = user_data.get_imu_data()[IMUDataFilterType.RAW]
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
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
            height_m = user_data.get_clinical_demo_data().get_height() / 100.0
            imu_data = user_data.get_imu_data(IMUDataFilterType.RAW)
            data = imu_data.get_triax_acc_data()
            data = np.array([data['vertical'], data['mediolateral'], data['anteroposterior']])
            data = data.T
            time = imu_data.get_time()
            fs = user_data.get_imu_metadata().get_sampling_frequency()
            if gait:
                results.append(pipeline_run.run_gait_pipeline(data, time, fs, height=height_m))
            else:
                results.append(pipeline_run.run_pipeline(data, time, fs, day_ends, height=height_m))
        return results

    def segment_data_walk(self, ds, gait_metric_names, day_ends, path_handler):
        skdh_output_path = path_handler.skdh_pipeline_results_folder
        # Run initial pass of SKDH on data
        pipeline_gen = SKDHPipelineGenerator()
        full_pipeline = pipeline_gen.generate_pipeline(skdh_output_path)
        full_pipeline_run = SKDHPipelineRunner(full_pipeline, gait_metric_names)
        skdh_input_metrics = self.generate_skdh_metrics(ds, day_ends, full_pipeline_run, False)
        skdh_pipeline_results_path = self.export_skdh_results(skdh_input_metrics, skdh_output_path)
        path_handler.skdh_pipeline_results_file = skdh_pipeline_results_path
        bout_ixs = self.get_walk_bout_ixs(skdh_input_metrics, ds, 20.0)
        if bout_ixs:
            walk_data, walk_time = self.get_walk_imu_data(bout_ixs, ds)
            # Create new dataset from the walking data segments
            walk_ds = self.gen_walk_ds(walk_data, walk_time, ds)
            self.preprocess_data(walk_ds)
        else:
            raise ValueError('FAILED TO SEGMENT DATA ALONG GAIT BOUTS')
        return walk_ds, skdh_input_metrics

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

    def gen_walk_ds(self, walk_data, walk_time, ds) -> Dataset:
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
        for walk_bout, time in zip(walk_data, walk_time):
            # Build a UserData object for the whole data
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
        time = np.array(imu_data.get_time())
        for bout_ix in bout_ixs:
            walk_data.append(acc_data[:, bout_ix[0]:bout_ix[1]])
            walk_time.append(time[bout_ix[0]:bout_ix[1]])
        return walk_data, walk_time

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
                if name not in [
                    'Bout Starts',
                    'Bout Duration',
                    'Bout Starts: mean',
                    'Bout Starts: std',
                    'Bout N: mean',
                    'Bout N: std'
                ]:
                    input_metrics.get_metric(name).append(val)
        for name, metric in custom_input_metrics.get_metrics().items():
            input_metrics.get_metric(name).extend(metric.get_value().tolist())
        final_metrics = InputMetrics()
        for name, metric in input_metrics.get_metrics().items():
            metric = np.array(metric)
            metric = metric[~np.isnan(metric)]
            metric_mean = metric.mean()
            final_metrics.set_metric(name, InputMetric(name, metric_mean))
        input_metrics.get_labels().extend(custom_input_metrics.get_labels())
        final_metrics.set_labels(input_metrics.get_labels())
        return final_metrics

    def initialize_input_metrics(self, skdh_input_metrics, custom_metric_names):
        input_metrics = InputMetrics()
        for name in custom_metric_names:
            input_metrics.set_metric(name, [])
        for name in skdh_input_metrics[0]['gait_metrics'].keys():
            if name not in ['Bout Starts',
                    'Bout Duration',
                    'Bout Starts: mean',
                    'Bout Starts: std',
                    'Bout N: mean',
                    'Bout N: std']:
                input_metrics.set_metric(name, [])
        input_metrics.set_labels([])
        return input_metrics

    def export_metrics(self, input_metrics: InputMetrics, path_handler: PathHandler):
        output_path = path_handler.ra_metrics_folder
        metric_file_name = 'model_input_metrics_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        full_path = os.path.join(output_path, metric_file_name)
        new_im = {}
        for name, metric in input_metrics.get_metrics().items():
            if isinstance(name, MetricNames):
                new_im[name.value] = metric.get_value()
            else:
                new_im[name] = metric.get_value()
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

    def load_data(self, path_handler: PathHandler):
        # Load in the assessment data file (contains device type of the IMU data)
        assess_info_path = path_handler.assessment_info_file
        assess_info = self.load_json_data(assess_info_path)
        data_type = assess_info['device_type']
        assess_uuid = assess_info['user_ID']
        # Load in the user data json file (contains demographic data of the user)
        clin_demo_path = path_handler.user_info_file
        clin_demo_data = self.load_json_data(clin_demo_path)
        cd_uuid = clin_demo_data['user_ID']
        if not cd_uuid == assess_uuid:
            raise ValueError(f'Assessment UUID does not match clinical demographic data UUID')
        # Load in the IMU data file based on type
        imu_data_path = path_handler.imu_data_file
        demo_path = path_handler.user_info_file
        # Build dataset objects
        return self.build_dataset(data_type, imu_data_path, clin_demo_data, demo_path)

    def build_dataset(self, data_type, imu_data_path, demo_data, demo_path):
        if data_type.lower() == 'mbientlab_metamotions':
            user_data: List[UserData] = MbientlabDatasetBuilder().build_single_user(imu_data_path, demo_data, demo_path)
        else:
            raise ValueError(f'Unknown IMU datatype provided {data_type}')
        dataset = Dataset(
            'mbientlab', [imu_data_path], [demo_path], user_data, {}
        )
        return dataset


class Model:
    def assess_fall_risk(self, model_path, scaler_path, metrics, path_handler: PathHandler):
        # TODO: verify classifier imported successfully
        risk_model = self.import_classifier(model_path, scaler_path)
        # Check correspondence between the input metrics and the metrics the model was trained on
        self.check_metric_correspond(metrics, risk_model)
        # TODO: verify metric formatting
        metrics = self.format_input_metrics_scaling(metrics)
        # TODO: verify metric scaling
        metrics = risk_model.scaler.transform(metrics)
        prediction = risk_model.make_prediction(metrics)[0]
        # TODO: verify prediction
        if prediction:
            if self.assess_elevated_risk(path_handler):
                prediction = 2
        results = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'prediction': prediction,
            'low-risk': 0,
            'moderate-risk': 1,
            'high-risk': 2
        }
        # TODO: results formatting and export
        file_path = self.export_results(results, path_handler)
        path_handler.ra_results_file = file_path
        return file_path, results

    def check_metric_correspond(self, metrics, risk_model):
        model_metrics = risk_model.model.feature_name()
        fafra_metrics = metrics.get_metric_names()
        fafra_metrics = [name.replace(':', '_') for name in fafra_metrics]
        fafra_metrics = [name.replace(' ', '_') for name in fafra_metrics]
        fafra_metrics = [name.replace('__', '_') for name in fafra_metrics]
        if model_metrics != fafra_metrics:
            raise ValueError('Lack of correspondence between FaFRA generated metrics and model training metrics: ' +
                             '\n' + f'FaFRA metric names: {fafra_metrics}'
                             '\n' + f'Model metric names: {model_metrics}')




    def assess_elevated_risk(self, path_handler):
        elevated_risk = False
        # Load in the gait metrics and check gait speed against accepted standard of 0.6 m/s
        results_file = path_handler.skdh_pipeline_results_file
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        gait_metrics = results_data['gait_metrics']
        gait_speed = gait_metrics['PARAM:gait speed: mean']
        if gait_speed < 0.6:
            elevated_risk = True
        return elevated_risk

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

    def export_results(self, results, path_handler):
        output_path = path_handler.ra_model_folder
        file_name = 'ra_results_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        full_path = os.path.join(output_path, file_name)
        with open(full_path, 'w') as f:
            json.dump(results, f)
        return full_path


def pipeline_test():
    fafra = FaFRA()
    ra_model_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_rcm_20220804-123836.pkl'
    ra_scaler_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_scaler_20220804-123836.bin'
    assessment_path = '/home/grainger/Desktop/risk_assessments/test_batch/batch_0000000000000000_YYYY_MM_DD/assessment_0000000000000000_YYYY_MM_DD/'
    ra = fafra.perform_risk_assessment(assessment_path, ra_model_path, ra_scaler_path)


def main():
    fafra = FaFRA()
    ra_model_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_rcm_20220804-123836.pkl'
    ra_scaler_path = '/home/grainger/Desktop/skdh_testing/ml_model/complete_im_models/model_2_2022_08_04/lgbm_skdh_ltmm_scaler_20220804-123836.bin'

    # ### FOR BATCH RUN ###
    # bridges_batch_001 = '/home/grainger/Desktop/risk_assessments/customer_Bridges/site_Bridges_Cornell_Heights/batch_0000000000000001_2022_11_11/'
    # for item in os.listdir(bridges_batch_001):
    #     if os.path.isdir(os.path.join(bridges_batch_001, item)):
    #         assessment_path = os.path.join(bridges_batch_001, item)
    #         ra = fafra.perform_risk_assessment(assessment_path, ra_model_path, ra_scaler_path)
    # #####################

    assessment_path = '/home/grainger/Desktop/risk_assessments/customer_Bridges/site_Bridges_Cornell_Heights/batch_0000000000000001_2022_11_11/assessment_0000000000000002_2022_11_11/'
    ra = fafra.perform_risk_assessment(assessment_path, ra_model_path, ra_scaler_path)


if __name__ == '__main__':
    pipeline_test()
