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

        results_gait = results['Gait']
        results_gait = {k: v.tolist() for k, v in results_gait.items() if type(v) is np.ndarray or type(v) is np.array}
        gk = [k for k,v in results_gait.items()]
        print(gk)
        # self.write_results_json(results_gait, 'gait_results', output_path)
        print('\n\n\n')

        results_act = results['ActivityLevelClassification']
        results_act = {k: v.tolist() for k, v in results_act.items() if type(v) is np.ndarray}
        # self.write_results_json(results_act, 'act_results', output_path)
        rk = [k for k,v in results_act.items()]
        print(rk)
        print('\n\n\n')
        print(results)
        pass

    def write_results_json(self, data, name, path):
        file = os.path.join(path, name + '.json')
        with open(file, 'w') as f:
            json.dump(data, f)

    def run_pipeline(self, data, time, pipeline):
        # final_ix = len(time) - 1
        # Start and stop index for the day (12AM day 0, 12AM day 1
        # testing
        day_ends = np.array([[3954300, 12594300]])
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
        pipeline.add(ActivityLevelClassification(), save_file=act_result_file, plot_file=act_plot_file)
        pipeline.add(ActivityLevelClassification(), save_file=act_result_file)
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
        print(data.shape)
        data = data.T
        print(data.shape)
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


