import time
from abc import abstractmethod
from typing import List, Any, Dict
from src.dataset_tools.params.motion_data import MotionData
from src.dataset_tools.params.sensor import Sensor
from src.motion_analysis.filters.motion_filters import MotionFilters


class MotionDataset:

    def __init__(self, name: str, path: str, file_format: Any, activity_ids: Any,
                 subject_data: Any, sampling_rate, sensor_data: Dict[str, Sensor]):
        self.name: str = name
        self.path: str = path
        self.file_format: Any = file_format
        self.activity_ids: Dict[str, Any] = activity_ids
        self.subject_data: Dict[str, Any] = subject_data
        self.sampling_rate: float = sampling_rate
        self.sensor_data: Dict[str, Sensor] = sensor_data
        self.motion_data: List[MotionData] = []


    @abstractmethod
    def read_dataset(self):
        pass

    @abstractmethod
    def write_dataset_to_csv(self, path):
        pass

    #TODO: Add in rest of get/set methods

    def get_name(self):
        return self.name

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_sensor_data(self):
        return self.sensor_data

    def get_motion_data(self) -> List[MotionData]:
        return self.motion_data

    def get_data(self, subject, activity, trial):
        motion_data = None
        for data in self.motion_data:
            data_subject_id = data.get_subject().get_subject_identifier()
            data_activity_code = data.get_activity().get_code()
            data_trial = data.get_trial()
            if data_subject_id == subject and data_activity_code == activity and data_trial == trial:
                motion_data = data
                break
        if motion_data:
            return motion_data
        else:
            raise ValueError(f'{subject}, {activity}, {trial}')

    def add_motion_data(self, motion_data: MotionData):
        self.motion_data.append(motion_data)

    def apply_lp_filter(self):
        filters = MotionFilters()
        for motion_data in self.get_motion_data():
            for tri_acc in motion_data.get_triaxial_accs():
                for acc in tri_acc.get_all_axes():
                    acc.set_lp_filtered_data(filters.apply_lpass_filter(acc.acceleration_data, self.sampling_rate))

    def apply_kalman_filter(self):
        filters = MotionFilters()
        total_md = len(self.get_motion_data())
        time_0 = time.time()
        for ix, motion_data in enumerate(self.get_motion_data()):
            print(ix)
            print(total_md)
            print(ix/total_md)
            print(f'runtime: {time.time()-time_0}')
            for tri_acc in motion_data.get_triaxial_accs():
                x_ax = tri_acc.get_x_axis()
                y_ax = tri_acc.get_y_axis()
                z_ax = tri_acc.get_z_axis()
                x_kf_filtered_data, y_kf_filtered_data, z_kf_filtered_data, unbiased_y_ax_kf_data = filters.apply_kalman_filter(x_ax, y_ax, z_ax, self.sampling_rate)
                x_ax.set_kf_filtered_data(x_kf_filtered_data)
                y_ax.set_kf_filtered_data(y_kf_filtered_data)
                z_ax.set_kf_filtered_data(z_kf_filtered_data)
                y_ax.set_unbiased_kf_filtered_data(unbiased_y_ax_kf_data)

    def calculate_first_derivative_data(self):
        filters = MotionFilters()
        for motion_data in self.get_motion_data():
            for tri_acc in motion_data.get_triaxial_accs():
                for acc in tri_acc.get_all_axes():
                    acc.set_first_derivative_data(filters.calculate_first_derivative(acc.acceleration_data, acc.time))




