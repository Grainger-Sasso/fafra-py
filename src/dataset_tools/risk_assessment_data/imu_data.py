import numpy as np


class IMUData:
    def __init__(self, activity_code, activity_description, v_acc_data,
                 ml_acc_data, ap_acc_data, yaw_gyr_data, pitch_gyr_data,
                 roll_gyr_data, time):
        # Activity code and activity code description
        self.activity_code: str = activity_code
        self.activity_description: str = activity_description
        # Vertical axis accelerometer data
        self.v_acc_data = np.array(v_acc_data)
        # Mediolateral axis accelerometer data
        self.ml_acc_data = np.array(ml_acc_data)
        # Anteroposterior axis accelerometer data
        self.ap_acc_data = np.array(ap_acc_data)
        # Vertical axis accelerometer data
        self.yaw_gyr_data = np.array(yaw_gyr_data)
        # Mediolateral axis accelerometer data
        self.pitch_gyr_data = np.array(pitch_gyr_data)
        # Anteroposterior axis accelerometer data
        self.roll_gyr_data = np.array(roll_gyr_data)
        # Arbitrary time axis
        self.time = time

    def get_activity_code(self):
        return self.activity_code

    def get_activity_description(self):
        return self.activity_description

    def get_acc_axis_data(self, axis):
        if axis == 'vertical':
            data = self.v_acc_data
        elif axis == 'mediolateral':
            data = self.ml_acc_data
        elif axis == 'anteroposterior':
            data = self.ap_acc_data
        else:
            raise ValueError(f'{axis} is not a valid axis')
        return data

    def get_gyr_axis_data(self, axis):
        if axis == 'yaw':
            data = self.yaw_gyr_data
        elif axis == 'pitch':
            data = self.pitch_gyr_data
        elif axis == 'roll':
            data = self.roll_gyr_data
        else:
            raise ValueError(f'{axis} is not a valid axis')
        return data

    def get_triax_acc_data(self):
        return {'vertical': self.v_acc_data,
                'mediolateral': self.ml_acc_data,
                'anteroposterior': self.ap_acc_data}

    def get_triax_gyr_data(self):
        return {'yaw': self.yaw_gyr_data,
                'pitch': self.pitch_gyr_data,
                'roll': self.roll_gyr_data}

    def get_all_data(self):
        return np.array([self.v_acc_data, self.ml_acc_data, self.ap_acc_data,
                         self.yaw_gyr_data, self.pitch_gyr_data, self.roll_gyr_data])

    def get_time(self):
        return self.time

    def set_acc_axis_data(self, axis, data):
        if axis == 'vertical':
            self.v_acc_data = data
        elif axis == 'mediolateral':
            self.ml_acc_data = data
        elif axis == 'anteroposterior':
            self.ap_acc_data = data
        else:
            raise ValueError(f'{axis} is not a valid axis')

    def set_gyr_axis_data(self, axis, data):
        if axis == 'yaw':
            self.yaw_gyr_data = data
        elif axis == 'pitch':
            self.pitch_gyr_data = data
        elif axis == 'roll':
            self.roll_gyr_data = data
        else:
            raise ValueError(f'{axis} is not a valid axis')
