import json
import numpy as np
import uuid
import os
import time
from scipy.stats import skew
from scipy.stats import kurtosis
from matplotlib import pyplot as plt

from src.motion_analysis.filters.motion_filters import MotionFilters
from src.risk_assessments.fall_risk_assessment import FallRiskAssessment
from src.motion_analysis.gait_analysis.gait_analyzer import GaitAnalyzer
from src.motion_analysis.gait_analysis.gait_analyzer_v2 import GaitAnalyzerV2
from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData
from src.dataset_tools.risk_assessment_data.imu_data_filter_type import IMUDataFilterType


class GaitSpeedValidator:
    def __init__(self):
        """
        Constructor of the GaitSpeedValidator class
        For the UIUC gait speed data, see for truth data:
        root folder -> Fixed speed data_Instrumented Treadmill
        """
        self.subj_gs_truth = {
            '101': {'CWT': 1.25, 'BS': 1.25},
            '102': {'CWT': 1.3, 'BS': 1.3},
            '103': {'CWT': 1.1, 'BS': 1.1},
            '104': {'CWT': 1.3, 'BS': 1.3},
            '105': {'CWT': 1.25, 'BS': 1.25},
            '106': {'CWT': 1.25, 'BS': 1.25},
            '107': {'CWT': 1.25, 'BS': 1.25},
            '108': {'CWT': 1.15, 'BS': 1.15},
            '109': {'CWT': 1.3, 'BS': 1.25},
            '110': {'CWT': 1.3, 'BS': 1.25},
            '111': {'CWT': 0.75, 'BS': 0.75},
            '112': {'CWT': 1.4, 'BS': 1.35},
            '113': {'CWT': 1.2, 'BS': 1.15},
            '114': {'CWT': 1.1, 'BS': 1.05},
            '115': {'CWT': 0.85, 'BS': 0.85},
            '201': {'CWT': 0.8, 'BS': 0.8},
            '202': {'CWT': 0.9, 'BS': 0.9},
            '203': {'CWT': 1.2, 'BS': 1.1},
            '204': {'CWT': 1.25, 'BS': 1.2},
            '205': {'CWT': 1.25, 'BS': 1.25},
            '206': {'CWT': 1.3, 'BS': 1.25},
            '207': {'CWT': 1.25, 'BS': 1.2},
            '208': {'CWT': 1.25, 'BS': 1.2},
            '209': {'CWT': 1.2, 'BS': 1.2},
            '210': {'CWT': 1.3, 'BS': 1.25},
            '211': {'CWT': 1.05, 'BS': 1.05},
            '212': {'CWT': 0.95, 'BS': 0.95},
            '213': {'CWT': 1.2, 'BS': 1.15},
            '214': {'CWT': 0.9, 'BS': 0.9},
            '215': {'CWT': 1.0, 'BS': 1.0},
            '216': {'CWT': 1.4, 'BS': 1.3},
            '217': {'CWT': 0.95, 'BS': 0.9},
            '219': {'CWT': 1.1, 'BS': 1.1},
            '220': {'CWT': 1.2, 'BS': 1.15},
            '221': {'CWT': 1.05, 'BS': 1.0},
            '222': {'CWT': 0.9, 'BS': 0.85},
            '223': {'CWT': 1.35, 'BS': 1.3},
            '224': {'CWT': 1.05, 'BS': 1.0},
            '225': {'CWT': 1.15, 'BS': 1.15},
            '226': {'CWT': 1.35, 'BS': 1.25},
            '227': {'CWT': 0.5, 'BS': 0.5},
            '228': {'CWT': 1.3, 'BS': 1.25},
            '229': {'CWT': 1.2, 'BS': 1.15},
            '230': {'CWT': 1.0, 'BS': 0.95},
            '231': {'CWT': 1.15, 'BS': 1.1},
            '301': {'CWT': 1.15, 'BS': 1.15},
            '302': {'CWT': 1.05, 'BS': 1.05},
            '304': {'CWT': 1.25, 'BS': 1.2},
            '305': {'CWT': 1.0, 'BS': 1.0},
            '306': {'CWT': 1.3, 'BS': 1.25},
            '307': {'CWT': 0.95, 'BS': 0.9},
            '309': {'CWT': 1.0, 'BS': 0.95},
            '310': {'CWT': 1.0, 'BS': 0.95},
            '311': {'CWT': 1.3, 'BS': 1.2}

        }
        self.filter = MotionFilters()

    def analyze_gait_speed_estimators(self, dataset_path, clinical_demo_path,
                                      segment_dataset, epoch_size):
        # Build dataset
        db = DatasetBuilder()
        dataset = db.build_dataset(dataset_path, clinical_demo_path,
                                   segment_dataset, epoch_size)
        # Run dataset through low-pass filter
        for user_data in dataset.get_dataset():
            self.apply_lpf(user_data, plot=False)
        # Run gait speed estimation on the dataset from both of the estimators
        gs_results_v1 = self.calc_gait_speeds_v1(dataset, version_num='1.0',
                                                hpf=False)
        gs_results_v2 = self.calc_gait_speeds_v2(dataset, version_num='2.0',
                                                hpf=False,
                                                check_against_truth=False,
                                                plot_gait_cycles=False)
        # Compare the results with truth values
        gs_results_1 = self.compare_gs_to_truth(gs_results_v1)
        gs_results_2 = self.compare_gs_to_truth(gs_results_v2)
        # Show performance comparison between the two estimators
        self.run_analyzer_comparison(gs_results_1, gs_results_2)

    def run_analyzer_comparison(self, gs_results_1, gs_results_2):
        """
        Comparison consists of displaying the occurrences of percentage
        differences for both estimators and some plotting
        """
        gs1 = np.array([r['estimate'] for r in gs_results_1])
        gs1_not_nan = [i['estimate'] for i in gs_results_1 if
                       not np.isnan(i['estimate'])]
        gs2 = np.array([r['estimate'] for r in gs_results_2])
        gs2_not_nan = [i['estimate'] for i in gs_results_2 if
                       not np.isnan(i['estimate'])]
        cwt_truth_values = [truth_val[1]['CWT'] for truth_val in
                            self.subj_gs_truth.items()]

        cwt_percent_diffs1 = [result['diff'] for result in gs_results_1 if
                              not np.isnan(result['diff'])]
        cwt_percent_diffs2 = [result['diff'] for result in gs_results_2 if
                              not np.isnan(result['diff'])]

        # Count the occurrences of percentage differences for both estimators
        diff_counts_1 = self.count_percent_diff_occurrences(gs_results_1)
        diff_counts_2 = self.count_percent_diff_occurrences(gs_results_2)

        # Print results to consol, starting with number of not nan results
        print(f'Number of diffs evaluated for 1 (not nan): {len(gs1_not_nan)}')
        print(f'Number of diffs evaluated for 2 (not nan): {len(gs2_not_nan)}')
        # Print the occurrences of percentage differences for both
        gs1_len = len(gs1)
        gs2_len = len(gs2)
        self.print_perc_diff_occurrences(diff_counts_1, gs1_len, '1')
        self.print_perc_diff_occurrences(diff_counts_2, gs2_len, '2')
        print('\n')
        # Print descriptive statistics for the results
        self.print_desc_stats(cwt_percent_diffs1, 'DIFFS1')
        self.print_desc_stats(cwt_percent_diffs2, 'DIFFS2')
        self.print_desc_stats(cwt_truth_values, 'TRUTH')
        self.print_desc_stats(gs1, 'GSE1')

        # fig, axes = plt.subplots(2)
        #
        # t1_not_nan = [i['truth'] for i in gs_results_1 if
        #               not np.isnan(i['truth'])]
        # d1_not_nan = [i['diff'] for i in gs_results_1 if
        #               not np.isnan(i['diff'])]
        # t2_not_nan = [i['truth'] for i in gs_results_2 if
        #               not np.isnan(i['truth'])]
        # d2_not_nan = [i['diff'] for i in gs_results_2 if
        #               not np.isnan(i['diff'])]
        # axes[0].plot(t1_not_nan, d1_not_nan, 'bo')
        # axes[0].plot(t1_not_nan, np.zeros(len(t1_not_nan)))
        # axes[1].plot(t2_not_nan, d2_not_nan, 'bo')
        # axes[1].plot(t2_not_nan, np.zeros(len(t2_not_nan)))
        #
        # plt.show()

        # bins = np.linspace(0.0, 2.0, 30)
        # fig, axes = plt.subplots(3)
        # axes[0].hist(cwt_truth_values, bins, alpha=1.0, label='truth')
        # axes[0].legend(loc='upper right')
        # axes[1].hist(gs1, bins, alpha=1.0, label='gs1')
        # axes[1].legend(loc='upper right')
        # axes[2].hist(gs2, bins, alpha=1.0, label='gs2')
        # axes[2].legend(loc='upper right')
        #
        # # plt.hist(cwt_truth_values, bins, alpha=1.0, label='truth')
        # # plt.hist(gs1, bins, alpha=0.33, label='gs1')
        # # plt.hist(gs2, bins, alpha=0.33, label='gs2')
        # # plt.legend(loc='upper right')
        # plt.show()

    def print_perc_diff_occurrences(self, diff_counts, results_len, ver_num):
        print(
            f'Percent of GSEV{ver_num} within 1% truth: {(diff_counts[0] / results_len) * 100}')
        print(
            f'Percent of GSEV{ver_num} within 3% truth: {diff_counts[1] / results_len * 100}')
        print(
            f'Percent of GSEV{ver_num} within 5% truth: {diff_counts[2] / results_len * 100}')
        print(
            f'Percent of GSEV{ver_num} within 10% truth: {diff_counts[3] / results_len * 100}')
        print('\n')


    def count_percent_diff_occurrences(self, results):
        count_1_percent = 0
        count_3_percent = 0
        count_5_percent = 0
        count_10_percent = 0
        for result in results:
            diff = result['diff']
            if not np.isnan(diff):
                if diff < 1.0:
                    count_1_percent += 1
                if diff < 3.0:
                    count_3_percent += 1
                if diff < 5.0:
                    count_5_percent += 1
                if diff < 10.0:
                    count_10_percent += 1
        diff_counts = [count_1_percent, count_3_percent, count_5_percent,
                         count_10_percent]
        return diff_counts

    def print_desc_stats(self, data, name):
        print(name)
        print(f'Min: {min(data)}')
        print(f'Max: {max(data)}')
        print(f'Mean: {np.mean(data)}')
        print(f'Median: {np.median(data)}')
        print(f'STD: {np.std(data)}')
        print(f'Skewness: {skew(data)}')
        if skew(data) > 0.0:
            print('Data skews to lower values')
        else:
            print('Data skews to higher values')
        print(f'Kurtosis {kurtosis(data)}')
        if skew(data) > 0.0:
            print('Tightly distributed')
        else:
            print('Widely distributed')
        print('\n')

    def calc_gait_speeds_v1(self, dataset: Dataset, write_out_results=False,
                            ouput_dir='', version_num='1.0', hpf=False,
                            max_com_v_delta=0.14, check_against_truth=False,
                            plot_gait_cycles=False):
        # Compare the results of the gait analyzer with truth values
        gs_results = []
        ga = GaitAnalyzer()
        for user_data in dataset.get_dataset():
            user_id = user_data.get_clinical_demo_data().get_id()
            trial = user_data.get_clinical_demo_data().get_trial()
            gait_speed = ga.estimate_gait_speed(user_data, hpf, max_com_v_delta,
                                                plot_gait_cycles)
            gs_results.append(dict({'id': user_id, 'trial': trial,
                                    'gait_speed': gait_speed}))
        if write_out_results:
            filename = 'gait_speed_estimator_results_v' + version_num + '_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
            output_path = os.path.join(ouput_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(gs_results, f)
        return gs_results

    def calc_gait_speeds_v2(self, dataset: Dataset, write_out_results=False,
                            ouput_dir='', version_num='1.0', hpf=False,
                            max_com_v_delta=0.14, check_against_truth=False,
                            plot_gait_cycles=False):
        # Compare the results of the gait analyzer with truth values
        gs_results = []
        ga = GaitAnalyzerV2()
        count = 0
        for user_data in dataset.get_dataset():
            user_id = user_data.get_clinical_demo_data().get_id()
            trial = user_data.get_clinical_demo_data().get_trial()
            gait_speed, fig = ga.estimate_gait_speed(user_data, hpf, max_com_v_delta,
                                                plot_gait_cycles)
            gs_results.append(dict({'id': user_id, 'trial': trial,
                                    'gait_speed': gait_speed}))
            if user_id in self.subj_gs_truth.keys() and not np.isnan(gait_speed):
                cwt_truth_value = self.subj_gs_truth[user_id]['CWT']
                # If the difference between estimate and truth exceeds 10%
                diff = (abs(cwt_truth_value - gait_speed)/cwt_truth_value) * 100.0
                if check_against_truth:
                    if 10.0 < diff < 15.0:
                        # Show figure and print out the diff, ID, and trial
                        print('Function returned Valid Results')
                        print(f'ID: {user_id}')
                        print(f'Trial: {trial}')
                        print(f'Diff: {diff}')
                        print(f'Truth: {cwt_truth_value}')
                        print(f'Estimate: {gait_speed}')
                        print('\n')
                        plt.show()
                        print('\n')
            else:
                # Print off the ID and trial
                print('Function returned NAN or unknown UUID')
                print(f'ID: {user_id}')
                print(f'Trial: {trial}')
                print(f'Estimate: {gait_speed}')
                print('\n')
            plt.close()
        if write_out_results:
            filename = 'gait_speed_estimator_results_v' + version_num + '_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
            output_path = os.path.join(ouput_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(gs_results, f)
        return gs_results

    def compare_gs_to_truth(self, gs_results):
        for result in gs_results:
            if result['id'] in self.subj_gs_truth.keys() and not np.isnan(result['gait_speed']):
                cwt_truth_value = self.subj_gs_truth[result['id']]['CWT']
                estimate = result['gait_speed']
                result['truth'] = cwt_truth_value
                result['estimate'] = estimate
                result['diff'] = (abs(cwt_truth_value - estimate)/cwt_truth_value * 100.0)
            else:
                result['truth'] = np.nan
                result['estimate'] = np.nan
                result['diff'] = np.nan
        return gs_results

    def compare_gse_to_baseline(self, gs_results, baseline_path):
        # Read in the baseline comparisons from baseline path
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        # Validate that the results of the gse match the baseline resutls
        erroneous_results = []
        for baseline_result in baseline_data:
            id = baseline_result['id']
            trial = baseline_result['trial']
            # Find corresponding result in gs results
            corr_gs_result = [gs for gs in gs_results if gs['id'] == id and gs['trial'] == trial]
            if len(corr_gs_result) == 1:
                corr_gs_val = corr_gs_result[0]['gait_speed']
                baseline_gs_val = baseline_result['gait_speed']
                if corr_gs_val != baseline_gs_val:
                    erroneous_results.append({'id': id, 'trial': trial, 'gs_result': corr_gs_val,
                                              'baseline_gs_value': baseline_gs_val})
            else:
                # Add this result to the list of erroneous results
                erroneous_results.append(f'No matching gs results for baseline {id}')
        return erroneous_results

    def apply_lpf(self, user_data, cutoff_freq=2.33, plot=False):
        imu_data: IMUData = user_data.get_imu_data()[IMUDataFilterType.RAW]
        samp_freq = user_data.get_imu_metadata().get_sampling_frequency()
        act_code = imu_data.get_activity_code()
        act_des = imu_data.get_activity_description()
        raw_acc_data = [
            user_data.get_imu_data(IMUDataFilterType.RAW).get_acc_axis_data('vertical'),
            user_data.get_imu_data(IMUDataFilterType.RAW).get_acc_axis_data('mediolateral'),
            user_data.get_imu_data(IMUDataFilterType.RAW).get_acc_axis_data('anteroposterior')
        ]
        lpf_data_all_axis = []
        for data in raw_acc_data:
            lpf_data_all_axis.append(
                self.filter.apply_lpass_filter(data, cutoff_freq, samp_freq))
        lpf_data_all_axis.extend([
            user_data.get_imu_data(IMUDataFilterType.RAW).get_gyr_axis_data('yaw'),
            user_data.get_imu_data(IMUDataFilterType.RAW).get_gyr_axis_data('pitch'),
            user_data.get_imu_data(IMUDataFilterType.RAW).get_gyr_axis_data('roll')
        ])
        lpf_imu_data = self._generate_imu_data_instance(lpf_data_all_axis,
                                                        samp_freq, act_code, act_des)
        user_data.imu_data[IMUDataFilterType.LPF] = lpf_imu_data
        if plot:
            # Plot the raw and filtered vertical data
            raw_v = raw_acc_data[0]
            filt_v = user_data.get_imu_data(IMUDataFilterType.LPF).get_acc_axis_data('vertical')
            time = np.linspace(0.0, len(raw_v)/samp_freq, len(raw_v))
            plt.plot(time, raw_v)
            plt.plot(time, filt_v)
            plt.show()

    def _generate_imu_data_instance(self, data, sampling_freq, act_code, act_des):
        v_acc_data = np.array(data[0])
        ml_acc_data = np.array(data[1])
        ap_acc_data = np.array(data[2])
        yaw_gyr_data = np.array(data[3])
        pitch_gyr_data = np.array(data[4])
        roll_gyr_data = np.array(data[5])
        time = np.linspace(0, len(v_acc_data) / int(sampling_freq),
                           len(v_acc_data))
        return IMUData(act_code, act_des, v_acc_data, ml_acc_data, ap_acc_data,
                       yaw_gyr_data, pitch_gyr_data, roll_gyr_data, time)


def main():
    val = GaitSpeedValidator()
    dataset_path = r'C:\Users\gsass\Documents\fafra\datasets\GaitSpeedValidation\GaitSpeedValidation\Hexoskin Binary Data files 2\Hexoskin Binary Data files'
    clinical_demo_path = 'N/A'
    segment_dataset = False
    epoch_size = 0.0
    val.analyze_gait_speed_estimators(dataset_path, clinical_demo_path, segment_dataset, epoch_size)


def run_comparison(val, gs_results):
    # Run the validation
    gs = np.array([r['gait_speed'] for r in gs_results])
    cwt_truth_values = [truth_val[1]['CWT'] for truth_val in val.subj_gs_truth.items()]
    bs_truth_values = [truth_val[1]['BS'] for truth_val in val.subj_gs_truth.items()]
    cwt_diffs, bs_diffs = val.compare_gs_to_truth(gs_results)
    percentages = [diff/val for diff, val in zip(cwt_diffs, cwt_truth_values)]
    pm = np.mean(percentages)
    print_desc_stats(cwt_diffs, 'DIFFS')
    print_desc_stats(cwt_truth_values, 'TRUTH')
    print_desc_stats(gs, 'GSE')
    print('a')


if __name__ == '__main__':
    main()
