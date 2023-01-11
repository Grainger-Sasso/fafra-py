import numpy as np

from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class MetricGenerator:
    def __init__(self):
        pass

    def gen_input_features(self, ds_path, clinic_demo_path,
                           output_path, segment_dataset, epoch_size):
        features = []
        # Load dataset
        ds = self.load_dataset(
            ds_path, clinic_demo_path,
            segment_dataset, epoch_size)
        # Characterize dataset
        self.characterize_dataset(ds, output_path)
        # Preprocess dataset
        self.preprocess_data(ds)
        # Generate SKDH metrics
        # Generate custom metrics
        # Format input metrics
        # Export input metrics
        return features

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
