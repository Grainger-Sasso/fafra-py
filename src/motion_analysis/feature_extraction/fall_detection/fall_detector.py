from src import MotionDataset


class FallDetector:

    def __init__(self):
        self.detection_methods = ['sucerquia']

    def detect_falls(self, motion_dataset: MotionDataset, method: str):
        if method == 'sucerquia' or 'Sucerquia':
            self.__detect_falls_sucerquia(motion_dataset)
        else:
            raise ValueError(f'Fall detection method provided, {method}, not available')

    def __detect_falls_sucerquia(self, motion_dataset: MotionDataset):
        fall_occurred = False
        fall_time = 0.0
        # Apply low pass filter
        motion_dataset.apply_lp_filter()
        # Apply derivative, J1
        motion_dataset.calculate_first_derivative_data()
        # Apply Kalman filter, J2
        motion_dataset.apply_kalman_filter()
        # Detect periodicity from Kalman filter
        # Multiply J1 * (J2)^2, score
        # For all times, t0-tf
        # If score > threshold and data is not periodic
        # A fall has occurred, label the time

        return fall_occurred, fall_time

