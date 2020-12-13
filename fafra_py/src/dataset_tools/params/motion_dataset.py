from abc import abstractmethod
from typing import List, Any, Dict
from fafra_py_legacy.src.dataset_tools.params.motion_data import MotionData
from fafra_py_legacy.src.dataset_tools.params.sensor import Sensor
from fafra_py_legacy.src.motion_analysis.filters.motion_filters import MotionFilters


class MotionDataset:

    def __init__(self, name: str, path: str, file_format: Any, activity_ids: Any,
                 subject_data: Any, sampling_rate,sensor_data: Dict[str, Sensor]):
        self.name: str = name
        self.path: str = path
        self.file_format: Any = file_format
        self.activity_ids: Dict[str, Any] = activity_ids
        self.subject_data: Dict[str, Any] = subject_data
        self.sampling_rate: float = sampling_rate
        self.sensor_data: Any = sensor_data
        self.motion_data: List[MotionData] = []
        self.filters: MotionFilters = MotionFilters()

    @abstractmethod
    def read_dataset(self):
        pass

    @abstractmethod
    def write_dataset_to_csv(self, path):
        pass

    def get_motion_data(self):
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

    def apply_lp_filter(self):
        for motion_data in self.get_motion_data():
            for tri_acc in motion_data.get_triaxial_accs():
                for acc in tri_acc.get_all_axes():
                    acc.set_lp_filtered_data(self.filters.apply_lpass_filter(acc.acceleration_data, self.sampling_rate))

    def apply_kalman_filter(self):
        for motion_data in self.get_motion_data():
            for tri_acc in motion_data.get_triaxial_accs():
                x_ax = tri_acc.get_x_axis()
                y_ax = tri_acc.get_y_axis()
                z_ax = tri_acc.get_z_axis()
                x_kf_filtered_data, y_kf_filtered_data, z_kf_filtered_data = self.filters.apply_kalman_filter(
                    x_ax, y_ax, z_ax)
                x_ax.set_kf_filtered_data(x_kf_filtered_data)
                y_ax.set_kf_filtered_data(y_kf_filtered_data)
                z_ax.set_kf_filtered_data(z_kf_filtered_data)

    def calculate_first_derivative_data(self):
        for motion_data in self.get_motion_data():
            for tri_acc in motion_data.get_triaxial_accs():
                for acc in tri_acc.get_all_axes():
                    acc.set_first_derivative_data(self.filters.calculate_first_derivative(acc.acceleration_data, acc.time))




