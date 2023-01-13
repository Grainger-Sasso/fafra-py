import numpy as np
from matplotlib import pyplot as plt
from typing import List

from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner


class MetricGenerator:
    def __init__(self):
        self.skdh_metric_names: List[str] = [
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

    def gen_input_metrics(self, ds_path, clinic_demo_path,
                           output_path, segment_dataset, epoch_size):
        features = []
        # Load dataset
        ds = self.load_dataset(
            ds_path, clinic_demo_path,
            segment_dataset, epoch_size)
        # Characterize dataset
        self.characterize_dataset(ds, output_path)
        # Preprocess dataset
        # self.preprocess_data(ds)
        # Generate SKDH metrics
        skdh_metrics, failed_trials = self.generate_skdh_metrics(ds)
        # TODO: Generate custom metrics
        # Format input metrics
        im = self.format_input_metrics(skdh_metrics, ds)
        # TODO: Export input metrics
        return im


    def format_input_metrics(self, skdh_input_metrics, ds):
        input_metrics: InputMetrics = self.initialize_input_metrics(skdh_input_metrics)
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
                    # Create function to get the user labels
            user_id = user_metrics['id']
            trial_id = user_metrics['trial']
            label = self.get_user_label(user_id, ds)
            input_metrics.labels.append(label)
            input_metrics.user_ids.append(user_id)
            input_metrics.trial_ids.append(trial_id)
        new_ims = InputMetrics()
        for name, metric in input_metrics.get_metrics().items():
            im = InputMetric(name, np.array(metric))
            new_ims.set_metric(name, im)
        new_ims.labels = input_metrics.labels
        new_ims.trial_ids = input_metrics.trial_ids
        new_ims.user_ids = input_metrics.user_ids
        return new_ims

    def get_user_label(self, id, ds: Dataset):
        # Map the user id to their fall risk label
        label = None
        for user_data in ds.get_dataset():
            user_id = user_data.get_clinical_demo_data().get_id()
            if user_id == id:
                label = user_data.get_clinical_demo_data().get_faller_status()
                break
        if label is not None:
            return label
        else:
            raise ValueError('Unable to get user fall status label')

    def initialize_input_metrics(self, skdh_input_metrics):
        input_metrics = InputMetrics()
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

    # def format_input_metrics(self, metrics, ds):
    #     # Format the metrics together with the labels for ML training
    #
    #     pass

    def generate_skdh_metrics(self, ds):
        pipeline_gen = SKDHPipelineGenerator()
        gait_pipeline = pipeline_gen.generate_gait_pipeline()
        gait_pipeline_run = SKDHPipelineRunner(
            gait_pipeline, self.skdh_metric_names)
        skdh_input_metrics, failed_trials = self.run_skdh_gait(
            ds, gait_pipeline_run)
        return skdh_input_metrics, failed_trials

    def run_skdh_gait(self, ds, gait_pipeline_run):
        results = []
        failed_trials = []
        for user_data in ds.get_dataset():
            height_m = user_data.get_clinical_demo_data().get_height()
            imu_data = user_data.get_imu_data(IMUDataFilterType.RAW)
            data = imu_data.get_triax_acc_data()
            data = np.array([
                    data['vertical'],
                    data['mediolateral'],
                    data['anteroposterior']
                ])
            data = data.T
            time = imu_data.get_time()
            sid = user_data.get_clinical_demo_data().get_id()
            trial = user_data.get_clinical_demo_data().get_trial()
            fs = user_data.get_imu_metadata().get_sampling_frequency()
            result = gait_pipeline_run.run_gait_pipeline(data, time, fs, height=height_m)
            if np.isnan(result['gait_metrics'][self.skdh_metric_names[0] + ': mean']):
                failed_trials.append({'id': sid, 'trial': trial})
            else:
                result['id'] = sid
                result['trial'] = trial
                results.append(result)
        return results, failed_trials

    def visualize_user_data(self, user_data, sid, trial):
        imu_data = user_data.get_imu_data(IMUDataFilterType.RAW)
        v_acc = imu_data.get_acc_axis_data('vertical')
        ml_acc = imu_data.get_acc_axis_data('mediolateral')
        ap_acc = imu_data.get_acc_axis_data('anteroposterior')
        time = imu_data.get_time()
        fig, ax = plt.subplots()
        ax.plot(time, v_acc)
        ax.plot(time, ml_acc)
        ax.plot(time, ap_acc)
        ax.set_title(str(sid) + ': ' + str(trial))
        plt.show()

    def characterize_dataset(self, ds, output_path):
        """
        Need to fill in the details here, basically the
        number of trials, number of segments,
        demographic breakdowns, etc. Needs to be done in-lin
        with the metric generation because it references
        transitory elements. Output results as JSON file
        :param ds:
        :return:
        """
        ###
        # TODO: Low priority, finish last
        #  Characterize dataset (for both seg and non-seg)
        #  number of trials/samples, total time available, etc see notes.
        #  Export to a file
        ###
        pass

    def preprocess_data(self, ds):
        for user_data in ds.get_dataset():
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

    def load_dataset(self, ds_path, clinic_demo_path,
                     segment_dataset, epoch_size):
        db = DatasetBuilder()
        dataset = db.build_dataset(ds_path, clinic_demo_path,
                                   segment_dataset, epoch_size)
        return dataset


def main():
    mg = MetricGenerator()
    ds_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/bin_data/subj_files/'
    clinic_demo_path = '/home/grainger/Desktop/datasets/UIUC_gaitspeed/participant_metadata/Data_CHI2021_Carapace.xlsx'
    output_path = ''
    mg.load_dataset(ds_path, clinic_demo_path, output_path)


if __name__ == "__main__":
    main()
