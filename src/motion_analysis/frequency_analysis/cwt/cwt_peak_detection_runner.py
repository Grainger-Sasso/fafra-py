import os
import time
import numpy as np
from typing import List

from src.motion_analysis.frequency_analysis.cwt.continuous_wavelet_transform import CWT
from src.motion_analysis.attitude_estimation.attitude_estimator import AttitudeEstimator
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.dataset_tools.dataset_builders.builder_instances.sisfall_dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


def main():
    """
    Function should give me a sense of how the CWT behaves on ADL data
    Want to return a bunch of images of peak detection results
    Look at a couple cases first as a sanity check, then rip the whole batch
    :return:
    """
    t0 = time.time()
    # Set the paths to the sisfall dataset
    path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\SisFall_csv\SisFall_elderly_csv'
    # Instantiate the dataset builder
    db = DatasetBuilder()
    # Build dataset
    dataset: Dataset = db.build_dataset(path, '', True, 8.0)
    t_dataset = time.time()
    print(str(t_dataset-t0))
    # Get instances of SiSt and StSi
    act_codes = ['D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13']
    user_data: List[UserData] = dataset.get_dataset()
    adl_dataset = [data for data in user_data if data.get_imu_data()[IMUDataFilterType.RAW].get_activity_code() in act_codes]
    # Instantiate CWT and other parameters
    cwt = CWT()
    # Scales to freq for mexh wavelet correspond as follows:
    # (125.0 -> 0.2Hz) and (12.5 ->2.0Hz)
    min_max_scales = [125.0, 12.5]
    samp_freq = 200.0
    samp_period = 1 / samp_freq
    num_scales = 100
    scales = np.linspace(min_max_scales[0], min_max_scales[1],
                         num_scales).tolist()
    output_dir = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\cwt\plots'
    # Iterate through the ADL dataset and run CWT + peak detection on the batch
    for data in adl_dataset:
        filename = os.path.split(os.path.splitext(data.get_imu_data_file_path())[0])
        v_acc_data = preprocess_data(data, samp_freq, 'remove_gravity')
        coeffs, freqs = cwt.apply_cwt(v_acc_data, scales, samp_period)
        coeff_sums = cwt.sum_coeffs(coeffs)
        peaks = cwt.detect_cwt_peaks(coeff_sums, samp_period)
        peak_ix = peaks[0]
        peak_values = peaks[1]['peak_heights']
        cwt.plot_cwt_results(coeffs, freqs, samp_period, coeff_sums, peak_ix,
                             peak_values, output_dir, filename)
    # Evaluate the results
    pass

def preprocess_data(user_data: UserData, samp_freq, type: str):
    # Apply LPF
    lpf_data(user_data, samp_freq)
    if type == 'mean_subtraction':
        v_acc_data = user_data.get_imu_data()[
            IMUDataFilterType.LPF].get_acc_axis_data('vertical')
        return v_acc_data - np.mean(v_acc_data)
    elif type == 'remove_gravity':
        # Run the attitude estimation and remove the gravity componenet from
        remove_gravity(user_data)

def lpf_data(user_data: UserData, samp_freq):
    mf = MotionFilters()
    imu_data: IMUData = user_data.get_imu_data()[IMUDataFilterType.RAW]
    act_code = imu_data.get_activity_code()
    all_raw_data = imu_data.get_all_data()
    lpf_data_all_axis = []
    for data in all_raw_data:
        lpf_data_all_axis.append(
            mf.apply_lpass_filter(data, 2, samp_freq))
    lpf_imu_data = generate_imu_data_instance(lpf_data_all_axis, samp_freq, act_code)
    user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data

def generate_imu_data_instance(data, samp_freq, act_code):
    v_acc_data = np.array(data[0])
    ml_acc_data = np.array(data[1])
    ap_acc_data = np.array(data[2])
    yaw_gyr_data = np.array(data[3])
    pitch_gyr_data = np.array(data[4])
    roll_gyr_data = np.array(data[5])
    time = np.linspace(0, len(v_acc_data) / int(samp_freq),
                       len(v_acc_data))
    return IMUData(act_code, v_acc_data, ml_acc_data, ap_acc_data,
                   yaw_gyr_data, pitch_gyr_data, roll_gyr_data, time)

def remove_gravity(user_data):
    att_est = AttitudeEstimator()
    # Estimate the angle between the z-axis and the x-y plane
    theta = att_est.estimate_attitude(user_data, False)
    # Use the angle estimation to remove the effects of gravity from the
    # vertical acceleration
    v_acc_data = user_data.get_imu_data()[
        IMUDataFilterType.LPF].get_acc_axis_data('vertical')
    # As theta approaches 90°, amount of gravity component removed increases
    # As theta approaches 0°, amount of gravity component removed decreases
    rm_g_v_acc_data = [(v_acc, theta) for v_acc, theta in zip(v_acc_data, theta)]




if __name__ == "__main__":
    main()
