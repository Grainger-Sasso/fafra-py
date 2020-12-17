from typing import List
from src.dataset_tools.motion_data.acceleration.acceleration import Acceleration


class TriaxialAcceleration:

    def __init__(self, name: str, x_axis: Acceleration, y_axis: Acceleration,
                 z_axis: Acceleration):
        # Name (sensor info, linear or angular, etc.)
        self.name: str = name
        # Lateral aspect (left-right)
        self.x_axis: Acceleration = x_axis
        # Anterior-posterior (front-back)
        self.y_axis: Acceleration = y_axis
        # Superior-inferior (top-bottom)
        self.z_axis: Acceleration = z_axis
        self.sensor_name = self.get_sensor_name()

    def get_x_axis(self) -> Acceleration:
        return self.x_axis

    def get_y_axis(self) -> Acceleration:
        return self.y_axis

    def get_z_axis(self) -> Acceleration:
        return self.z_axis

    def get_all_axes(self) -> List[Acceleration]:
        return [self.x_axis, self.y_axis, self.z_axis]

    def get_sensor_name(self):
        if self.x_axis.sensor.name == self.y_axis.sensor.name == self.z_axis.sensor.name:
            sensor_name = self.x_axis.sensor.name
            return sensor_name
        else:
            raise ValueError(f'Axes in {self.sensor_name} not from the same sensors \n '
                             f'X axis sensor: {self.x_axis.sensor.name} \n '
                             f'Y axis sensor: {self.y_axis.sensor.name} \n '
                             f'Z axis sensor: {self.z_axis.sensor.name} \n ')

