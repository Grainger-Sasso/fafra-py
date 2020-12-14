import numpy as np
from fafra_py.src.dataset_tools.motion_data.acceleration.acceleration import Acceleration
from fafra_py.src.dataset_tools.params.sensor import Sensor


class LinearAcceleration(Acceleration):

    def __init__(self, axis: str, anatomical_axis: str, linear_acceleration_data: np.array, time, sensor: Sensor):
        super().__init__(axis, anatomical_axis, linear_acceleration_data, time, sensor)

