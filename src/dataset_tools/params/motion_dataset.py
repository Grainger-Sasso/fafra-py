from abc import abstractmethod
from typing import List, Any, Dict
from src.dataset_tools.params.motion_data import MotionData
from src.dataset_tools.params.sensor import Sensor


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

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def add_motion_data(self, motion_data: MotionData):
        self.motion_data.append(motion_data)

    # def apply_lp_filter(self):
    #     for ix, motion_data in enumerate(self.get_motion_data()):
    #         motion_data.apply_lp_filter(self.sampling_rate)
    #
    # def apply_kalman_filter(self):
    #     for ix, motion_data in enumerate(self.get_motion_data()):
    #         motion_data.apply_kalman_filter(self.sampling_rate)
    #
    # def calculate_first_derivative_data(self):
    #     for ix, motion_data in enumerate(self.get_motion_data()):
    #         motion_data.calculate_first_derivative_data()
    #
    # def downsample_dataset(self, old_sampling_rate, new_sampling_rate):
    #     for ix, motion_data in enumerate(self.get_motion_data()):
    #         motion_data.downsample(old_sampling_rate, new_sampling_rate)




