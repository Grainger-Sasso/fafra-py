import json
import os
import pandas as pd
import numpy as np
from src.motion_analysis.fall_detection.sucerquia_fall_detector import SucerquiaFallDetector
from src.dataset_tools.params.motion_dataset import MotionDataset
from src.dataset_tools.params.motion_data import MotionData
from src.dataset_tools.params.sensor import Sensor
from src.dataset_tools.motion_data.acceleration.linear_acceleration.triaxial_linear_acceleration import TriaxialLinearAcceleration
from src.dataset_tools.motion_data.acceleration.angular_acceleration.triaxial_angular_acceleration import TriaxialAngularAcceleration
from src.dataset_tools.motion_data.acceleration.acceleration import Acceleration
from src.dataset_tools.params.subject import Subject
from src.dataset_tools.params.activity import Activity


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSucerquiaFallDetector:

    def read_json_file(self, path):
        with open(path) as f:
            return json.load(f)

    def test_detect_falls_in_motion_dataset(self):
        # Instantiate fall detector
        sfd = SucerquiaFallDetector(4.44)
        # Read in acceleration data
        acc_data = self.read_json_file(os.path.join(TEST_DIR, 'unit_test_data', 'acceleration_data.json'))
        # Generate a motion dataset to test
        # name: str, path: str, file_format: Any, activity_ids: Any, subject_data: Any, sampling_rate,
        # sensor_data: Dict[str, Sensor]
        test_sensor = Sensor('test_sensor', '', 200.0, 0.0, 0.0)
        motion_dataset = MotionDataset('test_dataset', '', '', {'', ''}, {'', ''}, 200.0, {'test_sensor': test_sensor})
        # Generate motion data object for an instance of fall data
        subject_f = Subject('id', 50, 150, 120.0, 'male')
        activity_f = Activity('test_fall', True, 'test fall', 1, 10)
        x_lin_acc_f = Acceleration('x', 'sag', np.array(acc_data['x_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        y_lin_acc_f = Acceleration('y', 'sag', np.array(acc_data['y_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        z_lin_acc_f = Acceleration('z', 'sag', np.array(acc_data['z_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        x_ang_acc_f = Acceleration('x', 'sag', np.array(acc_data['x_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        y_ang_acc_f = Acceleration('y', 'sag', np.array(acc_data['y_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        z_ang_acc_f = Acceleration('z', 'sag', np.array(acc_data['z_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        tri_lin_accs_f = [TriaxialLinearAcceleration('name', x_lin_acc_f, y_lin_acc_f, z_lin_acc_f)]
        tri_ang_accs_f = [TriaxialAngularAcceleration('name', x_ang_acc_f, y_ang_acc_f, z_ang_acc_f)]
        motion_data_fall = MotionData(subject_f, activity_f, '1', pd.DataFrame([]), tri_lin_accs_f, tri_ang_accs_f)
        # Generate motion data object for an instance of ADL data
        subject_f = Subject('id', 50, 150, 120.0, 'male')
        activity_f = Activity('test_adl', False, 'test adl', 1, 10)
        x_lin_acc_a = Acceleration('x', 'sag', np.array(acc_data['x_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        y_lin_acc_a = Acceleration('y', 'sag', np.array(acc_data['y_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        z_lin_acc_a = Acceleration('z', 'sag', np.array(acc_data['z_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        x_ang_acc_a = Acceleration('x', 'sag', np.array(acc_data['x_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        y_ang_acc_a = Acceleration('y', 'sag', np.array(acc_data['y_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        z_ang_acc_a = Acceleration('z', 'sag', np.array(acc_data['z_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        tri_lin_accs_a = [TriaxialLinearAcceleration('name', x_lin_acc_a, y_lin_acc_a, z_lin_acc_a)]
        tri_ang_accs_a = [TriaxialAngularAcceleration('name', x_ang_acc_a, y_ang_acc_a, z_ang_acc_a)]
        motion_data_adl = MotionData(subject_f, activity_f, '1', pd.DataFrame([]), tri_lin_accs_a, tri_ang_accs_a)
        motion_dataset.add_motion_data(motion_data_fall)
        motion_dataset.add_motion_data(motion_data_adl)
        dataset_fall_detections, fall_detection_verifications, dataset_fall_indices = \
            sfd.detect_falls_in_motion_dataset(motion_dataset)
        assert all(dataset_fall_detections == np.array([True, False]))
        assert all(fall_detection_verifications == np.array([True, True]))
        assert dataset_fall_indices == [1388, np.nan]

    def test_detect_falls_in_motion_data(self):
        # Instantiate fall detector
        sfd = SucerquiaFallDetector(4.0)
        # Read in acceleration data
        acc_data = self.read_json_file(os.path.join(TEST_DIR, 'unit_test_data', 'acceleration_data.json'))
        sampling_rate = 200.0
        # Generate a motion dataset to test
        # name: str, path: str, file_format: Any, activity_ids: Any, subject_data: Any, sampling_rate, sensor_data: Dict[str, Sensor]
        test_sensor = Sensor('test_sensor', '', 200.0, 0.0, 0.0)
        # Generate motion data object for an instance of fall data
        subject_f = Subject('id', 50, 150, 120.0, 'male')
        activity_f = Activity('test_fall', True, 'test fall', 1, 10)
        x_lin_acc_f = Acceleration('x', 'sag', np.array(acc_data['x_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        y_lin_acc_f = Acceleration('y', 'sag', np.array(acc_data['y_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        z_lin_acc_f = Acceleration('z', 'sag', np.array(acc_data['z_lin_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        x_ang_acc_f = Acceleration('x', 'sag', np.array(acc_data['x_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        y_ang_acc_f = Acceleration('y', 'sag', np.array(acc_data['y_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        z_ang_acc_f = Acceleration('z', 'sag', np.array(acc_data['z_ang_acc_f']), np.array(acc_data['time_f']),
                                   test_sensor)
        tri_lin_accs_f = [TriaxialLinearAcceleration('name', x_lin_acc_f, y_lin_acc_f, z_lin_acc_f)]
        tri_ang_accs_f = [TriaxialAngularAcceleration('name', x_ang_acc_f, y_ang_acc_f, z_ang_acc_f)]
        motion_data_fall = MotionData(subject_f, activity_f, '1', pd.DataFrame([]), tri_lin_accs_f, tri_ang_accs_f)
        # Generate motion data object for an instance of ADL data
        subject_f = Subject('id', 50, 150, 120.0, 'male')
        activity_f = Activity('test_adl', False, 'test adl', 1, 10)
        x_lin_acc_a = Acceleration('x', 'sag', np.array(acc_data['x_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        y_lin_acc_a = Acceleration('y', 'sag', np.array(acc_data['y_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        z_lin_acc_a = Acceleration('z', 'sag', np.array(acc_data['z_lin_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        x_ang_acc_a = Acceleration('x', 'sag', np.array(acc_data['x_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        y_ang_acc_a = Acceleration('y', 'sag', np.array(acc_data['y_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        z_ang_acc_a = Acceleration('z', 'sag', np.array(acc_data['z_ang_acc_a']), np.array(acc_data['time_a']),
                                   test_sensor)
        tri_lin_accs_a = [TriaxialLinearAcceleration('name', x_lin_acc_a, y_lin_acc_a, z_lin_acc_a)]
        tri_ang_accs_a = [TriaxialAngularAcceleration('name', x_ang_acc_a, y_ang_acc_a, z_ang_acc_a)]
        motion_data_adl = MotionData(subject_f, activity_f, '1', pd.DataFrame([]), tri_lin_accs_a, tri_ang_accs_a)
        fall_data_fall_detections, fall_detection_verifications_f, fall_data_fall_indices = \
            sfd.detect_falls_in_motion_data(motion_data_fall, sampling_rate, False)
        adl_data_fall_detections, fall_detection_verifications_a, adl_data_fall_indices = \
            sfd.detect_falls_in_motion_data(motion_data_adl, sampling_rate, False)
        assert fall_data_fall_detections == True
        assert fall_detection_verifications_f == True
        assert fall_data_fall_indices == 1388
        assert adl_data_fall_detections == False
        assert fall_detection_verifications_a == True
        assert np.isnan(adl_data_fall_indices)

def main():
    tester = TestSucerquiaFallDetector()
    tester.test_detect_falls_in_motion_data()

if __name__ == '__main__':
    main()


# "C:\git\fafra-py\tests\unit_tests\test_sucerquia_fall_detection.py"
