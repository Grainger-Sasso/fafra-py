import json
import numpy as np
import uuid
import os
import time
from uuid import uuid4
from scipy.stats import skew
from scipy.stats import kurtosis
from matplotlib import pyplot as plt
from pathlib import Path

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

    def get_cwt_truth_values(self, comparison_results):
        truth_vals = []
        for result in comparison_results:
            if result['id'] in self.subj_gs_truth.keys():
                truth_vals.append(self.subj_gs_truth[result['id']]['CWT'])
        return truth_vals

    def analyze_gait_speed_estimators(self, dataset_path, clinical_demo_path,
                                      segment_dataset, epoch_size, results_location, eval_percentages, write_out_results):
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
        gs_results_v2 = self.calc_gait_speeds_v2(dataset, eval_percentages, results_location)
        # Compare the results with truth values, this returns a comparison for
        # every estimate that has a corresponding truth value (including) est-
        # imates that are nan
        self.compare_analyzers(gs_results_v1, gs_results_v2, eval_percentages)

    def compare_analyzers(self, truth_comparisons_1, truth_comparisons_2, eval_percentages):
        """
        Comparison consists of displaying the occurrences of percentage
        differences for both estimators and some plotting
        """
        # Print the number of files that an estimate was made for
        print(f'[1] Number of estimates (including nan): {len(truth_comparisons_1)}')
        print(f'[2] Number of estimates (including nan): {len(truth_comparisons_2)}')
        print('\n')
        # Print the number of files that an estimate was made for where the
        # result was not nan
        est_not_nan_1 = [comp for comp in truth_comparisons_1 if
                         not np.isnan(comp['estimate'])]
        number_est_not_nan_1 = len(est_not_nan_1)
        est_not_nan_2 = [comp for comp in truth_comparisons_2 if
                         not np.isnan(comp['estimate'])]
        number_est_not_nan_2 = len(est_not_nan_2)
        print(f'[1] Number of estimates (NOT including nan): {number_est_not_nan_1}')
        print(f'[2] Number of estimates (NOT including nan): {number_est_not_nan_2}')
        print('\n')
        # Print the number of occurrences of percent differences below given values
        # Count the occurrences of percentage differences for both estimators
        diff_counts_1 = self.count_percent_diff_occurrences(truth_comparisons_1, eval_percentages)
        diff_counts_2 = self.count_percent_diff_occurrences(truth_comparisons_2, eval_percentages)
        self.print_perc_diff_occurrences(diff_counts_1, len(truth_comparisons_1), '1', eval_percentages)
        self.print_perc_diff_occurrences(diff_counts_2, len(truth_comparisons_2), '2', eval_percentages)

        # Print descriptive statistics for the results where results are not nan
        percent_diffs_1 = [comp['diff'] for comp in truth_comparisons_1 if
                           not np.isnan(comp['diff'])]
        percent_diffs_2 = [comp['diff'] for comp in truth_comparisons_2 if
                           not np.isnan(comp['diff'])]
        truth_values = self.get_cwt_truth_values(truth_comparisons_1)
        estamates_1 = [comp['estimate'] for comp in truth_comparisons_1 if
                           not np.isnan(comp['estimate'])]
        estamates_2 = [comp['estimate'] for comp in truth_comparisons_2 if
                       not np.isnan(comp['estimate'])]
        self.print_desc_stats(percent_diffs_1, 'DIFFS1')
        self.print_desc_stats(percent_diffs_2, 'DIFFS2')
        self.print_desc_stats(truth_values, 'TRUTH')
        self.print_desc_stats(estamates_1, 'GSE1')
        self.print_desc_stats(estamates_2, 'GSE2')

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

    def print_perc_diff_occurrences(self, diff_counts, results_len, ver_num, eval_percentages):
        for percentage, diff_count in zip(eval_percentages, diff_counts):
            print(
                f'Percent of GSEV{ver_num} within {str(percentage)}% truth: {(diff_count / results_len) * 100}')
        print('\n')

    def count_percent_diff_occurrences(self, results, eval_percentages):
        diff_counts = []
        for percentage in eval_percentages:
            count = 0
            for result in results:
                diff = result['diff']
                if not np.isnan(diff):
                    if diff < percentage:
                        count += 1
            diff_counts.append(count)
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
                            max_com_v_delta=0.14,
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
        truth_comparisons = self.compare_gs_to_truth_val(gs_results)
        if write_out_results:
            filename = 'gait_speed_estimator_results_v' + version_num + '_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
            output_path = os.path.join(ouput_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(gs_results, f)
        return truth_comparisons

    def calc_gait_speeds_v2(self, dataset: Dataset, eval_percentages,
                            results_location, write_out_results=False,
                            hpf=False, max_com_v_delta=0.14,
                            plot_gait_cycles=True):
        # TODO: fix the parameters that control plotting and exporting data, also fix how figures are created
        # Estimate the gait speed for every user/trial in dataset
        truth_comparisons = []
        ga = GaitAnalyzerV2()
        count = 0
        for user_data in dataset.get_dataset():
            user_id = user_data.get_clinical_demo_data().get_id()
            trial = user_data.get_clinical_demo_data().get_trial()
            gait_speed, fig = ga.estimate_gait_speed(user_data, hpf, max_com_v_delta,
                                                plot_gait_cycles)
            result = {'id': user_id, 'trial': trial, 'gait_speed': gait_speed}
            if user_id in self.subj_gs_truth.keys():
                comparison = self.compare_gs_to_truth_val(result)
                truth_comparisons.append(comparison)
                if write_out_results:
                    self.write_out_gs2_comparison_results(comparison, fig,
                                                          eval_percentages,
                                                          results_location)
            fig.close()
        return truth_comparisons

    def write_out_gs2_comparison_results(self, comparison, fig, eval_percentages,
                                         results_location):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        result_path = os.path.join(results_location, timestr)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        # For every percentage in the evaluation percentages
        for percentage in eval_percentages:
            # Create a folder for each evaluation percentage
            perc_dir_path = os.path.join(results_location, f'percentile_{str(percentage)}%')
            Path(perc_dir_path).mkdir(parents=True, exist_ok=True)
            # Write out every figure and JSON file that has a truth difference
            # less than the given evaluation percentage
            diff = comparison['diff']
            if not np.isnan(diff) and diff < percentage:
                user = comparison['id']
                trial = comparison['trial']
                uuid = str(uuid4())
                # Create filename and path for JSON data
                json_file_name = f'diff_{diff}_{user}_{trial}_{uuid}.json'
                json_file_path = os.path.join(perc_dir_path, json_file_name)
                # Create filename and path for figure data
                fig_file_name = f'fig_{diff}_{user}_{trial}_{uuid}.png'
                fig_file_path = os.path.join(perc_dir_path, fig_file_name)
                # Write out data to JSON file
                with open(json_file_path, 'w') as jf:
                    json.dump(comparison, jf)
                # Write out figure to PNG file
                plt.savefig(fig)

    def compare_gs_to_truth_val(self, result):
        # Compare the estimate to the truth value
        comparison = {}
        comparison['id'] = result['id']
        comparison['trial'] = result['trial']
        cwt_truth_value = self.subj_gs_truth[result['id']]['CWT']
        comparison['truth'] = cwt_truth_value
        # If the gs estimate is not a nan value, compare it to truth
        if not np.isnan(result['gait_speed']):
            estimate = result['gait_speed']
            comparison['estimate'] = estimate
            comparison['diff'] = (
                    abs(cwt_truth_value - estimate) / cwt_truth_value * 100.0)
        # Otherwise, add nan as the comparison value
        else:
            comparison['estimate'] = np.nan
            comparison['diff'] = np.nan
        return comparison


    # def compare_gs_to_truth_val(self, gs_results):
    #     truth_comparisons = []
    #     for result in gs_results:
    #         # If the result has a truth value in the list of truth values
    #         if result['id'] in self.subj_gs_truth.keys():
    #             # Compare the estimate to the truth value
    #             comparison = {}
    #             comparison['id'] = result['id']
    #             comparison['trial'] = result['trial']
    #             cwt_truth_value = self.subj_gs_truth[result['id']]['CWT']
    #             comparison['truth'] = cwt_truth_value
    #             # If the gs estimate is not a nan value, compare it to truth
    #             if not np.isnan(result['gait_speed']):
    #                 estimate = result['gait_speed']
    #                 comparison['estimate'] = estimate
    #                 comparison['diff'] = (
    #                         abs(cwt_truth_value - estimate) / cwt_truth_value * 100.0)
    #             # Otherwise, add nan as the comparison value
    #             else:
    #                 comparison['estimate'] = np.nan
    #                 comparison['diff'] = np.nan
    #             truth_comparisons.append(comparison)
    #     return truth_comparisons

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
    results_location = r'C:\Users\gsass\Documents\fafra\testing\gait_speed\evaluation_percentages'
    eval_percentages = [1.0, 3.0, 5.0, 10.0]
    write_out_results = True
    val.analyze_gait_speed_estimators(dataset_path, clinical_demo_path, segment_dataset, epoch_size, results_location, eval_percentages, write_out_results)


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
