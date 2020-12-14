import numpy as np
from fafra_py.src.dataset_tools.params.sensor import Sensor
from fafra_py.src.dataset_tools.motion_data.acceleration.acceleration import Acceleration


class AngularAcceleration(Acceleration):

    def __init__(self, axis: str, anatomical_axis: str, angular_acceleration_data: np.array, time, sensor: Sensor):
        super().__init__(axis, anatomical_axis, angular_acceleration_data, time, sensor)
