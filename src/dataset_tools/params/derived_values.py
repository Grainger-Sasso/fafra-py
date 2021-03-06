import numpy as np
from src.dataset_tools.params.motion_data import MotionData


class DerivedValues:
    # TODO: create capability of class to point to the specific data that derived values came from,
    #  add in getters and setters
    def __init__(self, origin: MotionData, derived_values):
        # Point the origin of the derived values to the motion data it came from
        self.origin = origin
        self.derived_values = derived_values


