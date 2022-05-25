import numpy as np

from src.motion_analysis.filters.motion_filters import MotionFilters
from src.fibion_mvp.fibion_dataset_builder import FibionDatasetBuilder
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.metric_generator import MetricGenerator
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType



class FibionFaFRA:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(self.dataset_path)
        self.filter = MotionFilters()
        self.mg = MetricGenerator()

    def load_dataset(self, dataset_path):
        fdb = FibionDatasetBuilder()
        return fdb.build_dataset(dataset_path, '')

    def perform_risk_analysis(self, input_metric_names=tuple(MetricNames.get_all_enum_entries())):
        self.preprocess_data()
        user_data = self.dataset.get_dataset()
        # Estimate subject gait speed
        gs = self.estimate_gait_speed()
        # Get subject step count from activity chart
        act_chart_sc = self.get_ac_step_count()
        # Estimate subject activity levels
        act_levels = self.estimate_activity_levels()
        # Get sleep disturbances from activity chart
        sleep_dis = self.get_ac_sleep_disturb()
        # Estimate subject fall risk
        fall_risk_score = self.estimate_fall_risk(input_metric_names)
        # Construct report


    def get_ac_sleep_disturb(self):
        pass

    def estimate_activity_levels(self):
        pass

    def estimate_gait_speed(self):
        pass

    def get_ac_step_count(self):
        pass

    def estimate_fall_risk(self, input_metric_names):
        input_metrics: InputMetrics = self.generate_risk_metrics(
            input_metric_names)

    def build_risk_report(self):
        pass

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
    dataset_path = r'C:\Users\gsass\Documents\Fall Project Master\datasets\fibion\io_test_data\bin'
    fib_fafra = FibionFaFRA(dataset_path)
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

