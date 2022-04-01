import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import animation

from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class AttitudeEstimator:
    def __init__(self):
        self.quiver_x = None
        self.quiver_y = None
        self.quiver_z = None
        self.quiver_ref_x = None
        self.quiver_ref_y = None
        self.quiver_ref_z = None
        self.fig = None
        self.ax = None

    def estimate_attitude(self, user_data: UserData, display_estimate: bool):
        """
        This method uses imu data and assumptions on the initial orientation of
        the sensor to calculate orientation (attitude) of the global frame
        relative to the sensor frame, represented as a quaternion, with the
        Madgwick algorithm. This representation of orientation in quaternions
        can be converted into vectors describing the axis of the sensor in 3-D
        Euclidian space.
        ### Assumptions, declarations ###:
        - Orientation of the trunk assumed to be that same as the orientation
          of the sensor
        - The three vectors of the global reference frame, [x, y, z] map to
          body/sensor frame as such:
            [x, y, z] -> [ml, ap, v]
        - Initial orientation of sensor is upright (sensor z-axis is parallel
          with global z-axis).
        - Only force acting on sensor at time t=0 is gravity
        ### Ref ###
        - http://www1.udel.edu/biolohttp://www1.udel.edu/biology/rosewc/kaap686/reserve/kalman_filtering/madgwick_etal_ieee_icrr_2011.pdfgy/rosewc/kaap686/reserve/kalman_filtering/madgwick_etal_ieee_icrr_2011.pdf
        - https://prgaero.github.io/Reports/p1b/semenovilya.pdf
        :return:
        """
        # TODO: Convert the gyr data from deg/s to rads/s
        # Initialize the global reference frame
        # Initialize sensor attitude quaternion state with global ref z-axis
        # Initialize first sensor attitude quaternion states
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])
        # Initialize list of all sensor attitude quaternion states
        quat_all_t = [quat_0]
        # Initialize Madgwick parameters
        # Accelerometer error param
        beta = 0.1
        # Accelerometer weight param
        gamma = 0.1
        # Compute change in global attitude relative to sensor attitude
        # for all imu data samples (all time, t)
        # Retrieve and reformat the imu data
        imu_data = user_data.get_imu_data(IMUDataFilterType.LPF).get_all_data()
        acc_data = imu_data[0:3].T
        gyr_data = imu_data[3:].T
        gyr_data = self.convert_deg_rad(gyr_data)
        acc_data, gyr_data = self.order_data(acc_data, gyr_data)
        sampling_rate = user_data.get_imu_metadata().get_sampling_frequency()
        for acc_t, gyr_t in zip(acc_data, gyr_data):
            acc_t = np.array(acc_t)
            # Get previous sensor attitude quaternion state
            quat_0 = quat_all_t[-1]
            # Estimate the rate of change of the global frame relative to
            # sensor frame
            s_omega = [0.0]
            s_omega.extend(gyr_t.tolist())
            quat_dot_omega = self.quat_mult(0.5 * quat_0, s_omega)
            # Compute min error between previous sensor attitude quaternion and
            # gravitational component in accelerometer data
            min_grav_err = self.compute_min_grav_error(quat_0, acc_t)
            mag_min_grav_err = np.linalg.norm(min_grav_err)
            quat_a = -1.0 * beta * (min_grav_err/mag_min_grav_err)
            gyr_factor = (1 - gamma) * (1 / sampling_rate)
            gyr_term = gyr_factor * np.array(quat_dot_omega)
            acc_term = gamma * quat_a
            quat_t = quat_0 + gyr_term + acc_term
            # Normalize sensor attitude quaternion state
            quat_t = self.norm(quat_t)
            quat_all_t.append(quat_t)
        orientation = self.convert_quat_to_3d(quat_all_t)
        # Derive the angle between the z (vertical) axis and the xy (ml-ap) plane
        theta = self.calc_vert_angle(orientation)
        if display_estimate:
            self.display_vectors(orientation)
        return theta

    def convert_deg_rad(self, gyr_data):
        return np.array([i * (math.pi/180) for i in gyr_data])

    def order_data(self, acc_data, gyr_data):
        # Convert the data from the format of ((v, ml, ap),(yaw, pitch, roll))
        # to ((ml, ap, v), (pitch, roll, yaw))
        acc_v = acc_data.T[0]
        acc_ml = acc_data.T[1]
        acc_ap = acc_data.T[2]
        ordered_acc_data = np.array([acc_ml, acc_ap, acc_v]).T
        gyr_v = gyr_data.T[0]
        gyr_ml = gyr_data.T[1]
        gyr_ap = gyr_data.T[2]
        ordered_gyr_data = np.array([gyr_ml, gyr_ap, gyr_v]).T
        return ordered_acc_data, ordered_gyr_data

    def convert_quat_to_3d(self, quat_all_t):
        init_orient = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        orient_all_t = [init_orient]
        for quat in quat_all_t:
            orient_t = []
            for v in init_orient:
                quat_v = [0.0]
                quat_v.extend(v)
                quat_c = self.get_quat_conjugate(quat)
                v_t = self.quat_mult(self.quat_mult(quat, quat_v), quat_c)[1:]
                orient_t.append(v_t)
            orient_all_t.append(orient_t)
        return orient_all_t

    def rotate_3d(self, ortn, quat):
        ornt_rot = []
        for v in ortn:
            quat_v = [0.0]
            quat_v.extend(v)
            quat_c = self.get_quat_conjugate(quat)
            v_t = self.quat_mult(self.quat_mult(quat, quat_v), quat_c)[1:]
            ornt_rot.append(v_t)
        return ornt_rot

    def get_quat_conjugate(self, quat):
        w, x, y, z = quat
        return [w, -x, -y, -z]

    def compute_min_grav_error(self, quat, a):
        quat1, quat2, quat3, quat4 = quat
        ax, ay, az = a
        m1 = np.array([[-2.0*quat3, 2.0*quat2, 0.0],
                       [2.0*quat4, 2.0*quat1, -4.0*quat2],
                       [-2.0*quat1, 2.0*quat4, -4.0*quat3],
                       [2.0*quat2, 2.0*quat3, 0.0]])
        m2 = np.array([[2*(quat2*quat4-quat1*quat3) - ax],
                       [2*(quat1*quat2-quat3*quat4) - ay],
                       [2*(0.5 - quat2**2 - quat3**2) - az]])
        return np.matmul(m1, m2).T.flatten()

    def norm(self, quat):
        mag = np.linalg.norm(quat)
        return np.array([i / mag for i in quat])

    def norm_all_quat(self, all_quat_t):
        return np.array([self.norm(i) for i in all_quat_t])

    def quat_mult(self, quat1, quat2):
        """
        Reference for current implementation
        https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
        https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion/42180896#42180896
        :param quat1:
        :param quat2:
        :return:
        """
        w1, x1, y1, z1 = quat1
        w2, x2, y2, z2 = quat2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return [w, x, y, z]

    def calc_vert_angle(self, orientation):
        # Returns angle between z axis of sensor and xy plane
        z_axis_mag = np.array([np.linalg.norm(ortn[2]) for ortn in orientation])
        z_component = np.array([ortn[2][2] for ortn in orientation])
        # return np.arcsin(z_component/z_axis_mag) * (360/(2*np.pi))
        return np.arcsin(z_component / z_axis_mag)

    def display_vectors(self, orientation):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))
        self.quiver_ref_x = self.ax.quiver(*(0,0,0,1,0,0), color='lightcoral')
        self.quiver_ref_y = self.ax.quiver(*(0,0,0,0,1,0), color='limegreen')
        self.quiver_ref_z = self.ax.quiver(*(0,0,0,0,0,1), color='cornflowerblue')
        # Initialize the quiver with the first orientation vector
        ortn_0 = orientation.pop(0)
        ortn_0_x = ortn_0[0]
        ortn_0_y = ortn_0[1]
        ortn_0_z = ortn_0[2]
        self.quiver_x = self.ax.quiver(*self.get_arrow(ortn_0_x), color='r')
        self.quiver_y = self.ax.quiver(*self.get_arrow(ortn_0_y), color='g')
        self.quiver_z = self.ax.quiver(*self.get_arrow(ortn_0_z), color='b')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        ani = animation.FuncAnimation(self.fig, self.update, frames=orientation,
                            interval=0.1)
        # f = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\animations\test.gif'
        # ani.save(f)
        plt.show()

    def get_arrow(self, ortn):
        return 0, 0, 0, ortn[0], ortn[1], ortn[2]

    def update(self, ortn):
        self.quiver_x.remove()
        self.quiver_y.remove()
        self.quiver_z.remove()
        ortn_0_x = ortn[0]
        ortn_0_y = ortn[1]
        ortn_0_z = ortn[2]
        self.quiver_x = self.ax.quiver(*self.get_arrow(ortn_0_x), color='r')
        self.quiver_y = self.ax.quiver(*self.get_arrow(ortn_0_y), color='g')
        self.quiver_z = self.ax.quiver(*self.get_arrow(ortn_0_z), color='b')
