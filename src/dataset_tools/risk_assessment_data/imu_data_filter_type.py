from enum import Enum


class IMUDataFilterType(Enum):
    RAW = 'raw'
    LPF = 'low-pass_filtered'
    KF = 'kalman_filtered'
    ATTITUDE_ESTIMATION = 'attitude_estimation'

    def get_name(self):
        # Self is the member here
        return self.name

    def get_value(self):
        # Self is the member here
        return self.value

    @classmethod
    def get_all_names(cls):
        return [metric_name.name for metric_name in cls]

    @classmethod
    def get_all_values(cls):
        return [metric_name.value for metric_name in cls]
