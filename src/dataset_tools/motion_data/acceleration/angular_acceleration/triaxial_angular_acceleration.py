from src.dataset_tools.motion_data.acceleration.triaxial_acceleration import TriaxialAcceleration
from src.dataset_tools.motion_data.acceleration.angular_acceleration.angular_acceleration import AngularAcceleration


class TriaxialAngularAcceleration(TriaxialAcceleration):

    def __init__(self, name, x_axis: AngularAcceleration, y_axis: AngularAcceleration,
                 z_axis: AngularAcceleration):
        super().__init__(name, x_axis, y_axis, z_axis)

