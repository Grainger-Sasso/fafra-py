import numpy as np

from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.fibion_mvp.skdh_pipeline import SKDHPipelineGenerator, SKDHPipelineRunner


class ModelTrainer:
    def __init__(self, dataset_path, clinical_demo_path, segment_dataset, epoch_size , custom_metric_names):
        self.dataset_path = dataset_path
        self.clinical_demo_path = clinical_demo_path
        self.segment_dataset = segment_dataset
        self.epoch_size = epoch_size
        self.dataset = self.build_dataset()
        self.custom_metric_names = custom_metric_names

    def generate_model(self, skdh_output_path, model_output_path, file_name):
        # Preprocess data
        self.preprocess_data()
        # Generate custom metrics
        custom_input_metrics: InputMetrics = self.generate_custom_metrics()
        print(custom_input_metrics)
        # Generate SKDH metrics
        # skdh_input_metrics = self.generate_skdh_metrics(skdh_output_path)
        # Format input metrics
        # input_metrics = self.format_input_metrics(custom_input_metrics, skdh_input_metrics)
        # Train model on input metrics
        # Export model, scaler
        # Export traning data
        pass

    def preprocess_data(self):
        for user_data in self.dataset.get_dataset():
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

    def build_dataset(self):
        db = DatasetBuilder()
        return db.build_dataset(
            self.dataset_path,
            self.clinical_demo_path,
            self.segment_dataset,
            self.epoch_size
        )

    def generate_custom_metrics(self) -> InputMetrics:
        mg = MetricGenerator()
        return mg.generate_metrics(
            self.dataset.get_dataset(),
            self.custom_metric_names
        )

    def generate_skdh_metrics(self, output_path):
        pipeline_gen = SKDHPipelineGenerator()
        pipeline = pipeline_gen.generate_pipeline(output_path)
        pipeline_run = SKDHPipelineRunner(pipeline)
        results = []
        for user_data in self.dataset.get_dataset():
            # Get the data from the user data in correct format
            # Get the time axis from user data
            # Get sampling rate
            # Generate day ends for the time axes
            data = []
            time = []
            fs = 0.0
            day_ends = []
            results.append(pipeline_run.run_pipeline(data, time, fs, day_ends))
        return results

    def format_input_metrics(self, custom_input_metrics: InputMetrics,
                             skdh_input_metrics):
        custom_metrics = custom_input_metrics.get_metric_matrix()
        custom_metrics = np.reshape(custom_metrics, (1, -1))


def main():
    dp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/mini'
    cdp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    seg = False
    epoch = 0.0
    metric_names = tuple(
        [
            MetricNames.AUTOCORRELATION,
            MetricNames.FAST_FOURIER_TRANSFORM,
            MetricNames.MEAN,
            MetricNames.ROOT_MEAN_SQUARE,
            MetricNames.STANDARD_DEVIATION,
            MetricNames.SIGNAL_ENERGY,
            MetricNames.COEFFICIENT_OF_VARIANCE,
            MetricNames.ZERO_CROSSING,
            MetricNames.SIGNAL_MAGNITUDE_AREA
        ]
    )
    mt = ModelTrainer(dp, cdp, seg, epoch, metric_names)
    mt.generate_model('','','')
    print('yup')


if __name__ == '__main__':
    main()

