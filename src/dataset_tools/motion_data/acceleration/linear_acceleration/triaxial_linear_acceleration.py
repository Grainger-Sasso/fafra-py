from src.dataset_tools.motion_data import TriaxialAcceleration
from src.dataset_tools.motion_data import LinearAcceleration


class TriaxialLinearAcceleration(TriaxialAcceleration):

    def __init__(self, name, x_axis: LinearAcceleration, y_axis: LinearAcceleration,
                 z_axis: LinearAcceleration):
        super().__init__(name, x_axis, y_axis, z_axis)
