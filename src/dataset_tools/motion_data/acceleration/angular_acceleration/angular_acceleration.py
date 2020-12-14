import numpy as np
from src import Sensor
from src.dataset_tools.motion_data import Acceleration


class AngularAcceleration(Acceleration):

    def __init__(self, axis: str, anatomical_axis: str, angular_acceleration_data: np.array, time, sensor: Sensor):
        super().__init__(axis, anatomical_axis, angular_acceleration_data, time, sensor)
