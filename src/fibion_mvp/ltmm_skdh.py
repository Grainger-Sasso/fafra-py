import os
import json
import time
import numpy as np
import pandas as pd
import calendar
import bisect
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import parser, tz
from skdh import Pipeline
from skdh.preprocessing import DetectWear
from skdh.gait import Gait
from skdh.sleep import Sleep
from skdh.activity import ActivityLevelClassification
from skdh.activity import TotalIntensityTime
import wfdb
import joblib


from src.dataset_tools.dataset_builders.builder_instances.ltmm_dataset_builder import DatasetBuilder


fs = 100.0


class LTMM_SKDH:
    def run_ltmm_skdh(self, data_path, output_path):
        # Read in LTMM data into numpy array
        data, time = self.read_ltmm_file(data_path)
        # Generate SKDH pipeline
        pipeline = self.generate_pipeline(output_path)
        # Run data through the pipeline
        results = self.run_pipeline(data, time, pipeline)

        gait_metric_names = [
            'PARAM:gait speed',
            'BOUTPARAM:gait symmetry index',
            'PARAM:cadence',
            'Bout Steps',
            'Bout Duration',
            'Bout N'
        ]
        gait_metrics = self.parse_results(results, 'Gait', gait_metric_names)
        gait_metrics = self.parse_gait_metrics(gait_metrics)

        act_metric_names = [
            'wake sed 5s epoch [min]',
            'wake light 5s epoch [min]',
            'wake mod 5s epoch [min]',
            'wake vig 5s epoch [min]',
            'wake sed avg duration',
        ]
        act_metrics = self.parse_results(results, 'ActivityLevelClassification', act_metric_names)

        sleep_metric_names = [
            'average sleep duration',
            'total sleep time',
            'percent time asleep',
            'number of wake bouts',
            'sleep average hazard',
        ]
        sleep_metrics = self.parse_results(results, 'Sleep', sleep_metric_names)

        for key, value in gait_metrics.items():
            print(key + ': ' + str(value))
        print('\n\n')
        for key, value in act_metrics.items():
            print(key + ': ' + str(value))
        print('\n\n')
        for key, value in sleep_metrics.items():
            print(key + ': ' + str(value))
        print(gait_metrics)

        tot_time_s = len(time) / fs
        print(tot_time_s)
        print(tot_time_s/(60 * 60))

    def parse_gait_metrics(self, gait_metrics):
        # Mean and std of all
        # Total of steps and duration
        new_gait_metrics = {}
        for name, metric_arr in gait_metrics.items():
            # Remove nans from the array
            new_metric = metric_arr[~np.isnan(metric_arr)]
            # Take average of the array
            metric_mean = new_metric.mean()
            metric_std = new_metric.std()
            # Replace value in metric dict
            new_gait_metrics[name + ': mean'] = metric_mean
            new_gait_metrics[name + ': std'] = metric_std

        step_sum, duration_sum = self.parse_bouts(gait_metrics)
        new_gait_metrics['Bout Steps: sum'] = step_sum
        new_gait_metrics['Bout Duration: sum'] = duration_sum
        return new_gait_metrics

    def parse_bouts(self, gait_metrics):
        bout_steps = []
        bout_durs = []
        bout_n = gait_metrics['Bout N']
        current_ix = 0
        for ix, bout in enumerate(bout_n):
            if bout > current_ix:
                current_ix = bout
                bout_steps.append(gait_metrics['Bout Steps'][ix])
                bout_durs.append(gait_metrics['Bout Duration'][ix])
        step_sum = np.array(bout_steps).sum()
        duration_sum = np.array(bout_durs).sum()
        return step_sum, duration_sum

    def parse_results(self, results, results_type, metric_names):
        results_act = results[results_type]
        act_metrics = {}
        for name in metric_names:
            if name in results_act.keys():
                act_metrics[name] = results_act[name]
            else:
                raise ValueError(f'{name} not valid activity metric name: {results_act.keys()}')
        return act_metrics

    def write_results_json(self, data, name, path):
        file = os.path.join(path, name + '.json')
        with open(file, 'w') as f:
            json.dump(data, f)

    def run_pipeline(self, data, time, pipeline):
        # final_ix = len(time) - 1
        # Start and stop index for the day (12AM day 0, 12AM day 1
        # day_ends = np.array([[3954300, 12594300]])
        day_ends = np.array([[2514300, 11154300]])
        data = np.ascontiguousarray(data)
        return pipeline.run(time=time, accel=data, fs=fs, height=1.77, day_ends={(12, 24): day_ends})

    def generate_pipeline(self, output_path):
        pipeline = Pipeline()
        pipeline.add(DetectWear())
        gait_result_file = os.path.join(output_path, 'gait_results.csv')
        gait_plot_file = os.path.join(output_path, 'gait_plot.pdf')
        pipeline.add(Gait(), save_file=gait_result_file, plot_file=gait_plot_file)
        act_result_file = os.path.join(output_path, 'activity_results.csv')
        act_plot_file = os.path.join(output_path, 'activity_plot.pdf')
        act = ActivityLevelClassification(cutpoints='esliger_lumbar_adult')
        # act.add(TotalIntensityTime(level='sed', epoch_length=60, cutpoints='esliger_lumbar_adult'))
        # pipeline.add(act, save_file=act_result_file, plot_file=act_plot_file)
        pipeline.add(act, save_file=act_result_file)
        sleep_result_file = os.path.join(output_path, 'sleep_results.csv')
        sleep_plot_file = os.path.join(output_path, 'sleep_plot.pdf')
        pipeline.add(Sleep(day_window=(12, 24)), save_file=sleep_result_file, plot_file=sleep_plot_file)
        return pipeline

    def read_ltmm_file(self, path):
        wfdb_record = wfdb.rdrecord(path)
        data = np.array(wfdb_record.p_signal)
        data = np.float16(data)
        v_acc_data = np.array(data.T[0])
        ml_acc_data = np.array(data.T[1])
        ap_acc_data = np.array(data.T[2])
        data = np.array([v_acc_data, ml_acc_data, ap_acc_data])
        data = data.T
        time = np.linspace(1657299657.0, (len(v_acc_data) / int(fs)) + 1657299657.0,
                           len(v_acc_data))
        return data, time


def main():
    data_path = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/CO001'
    output_path = '/home/grainger/Desktop/skdh_testing/ltmm/'

    # whole_ds = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/'
    # cdp = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/ClinicalDemogData_COFL.xlsx'
    # db = DatasetBuilder()
    # db.build_dataset(whole_ds, cdp, False, 0.0)

    ltmm_skdh = LTMM_SKDH()
    ltmm_skdh.run_ltmm_skdh(data_path, output_path)


if __name__ == '__main__':
    main()


