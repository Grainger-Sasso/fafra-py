import os
import time
import numpy as np
import pandas as pd
import calendar
import bisect
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import parser, tz
from skdh import Pipeline
from skdh.gait import Gait
from skdh.sleep import Sleep
from skdh.activity import ActivityLevelClassification
import wfdb
import joblib


fs = 100.0

class LTMM_SKDH:
    def run_ltmm_skdh(self, data_path, output_path):
        # Read in LTMM data into numpy array
        data, time = self.read_ltmm_file(data_path)
        # Generate SKDH pipeline
        pipeline = self.generate_pipeline(output_path)
        # Run data through the pipeline
        self.run_pipeline(data, time, pipeline)
        pass

    def run_pipeline(self, data, time, pipeline):
        pipeline.run(time=time, accel=data, fs=fs)

    def generate_pipeline(self, output_path):
        pipeline = Pipeline()
        gait_file = os.path.join(output_path, 'gait_results.csv')
        pipeline.add(Gait(), save_file=gait_file)
        act_file = os.path.join(output_path, 'activity_results.csv')
        pipeline.add(ActivityLevelClassification(), save_file=act_file)
        sleep_file = os.path.join(output_path, 'sleep_results.csv')
        pipeline.add(Sleep(), save_file=sleep_file)
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
        time = np.linspace(0, len(v_acc_data) / int(fs),
                           len(v_acc_data))
        return data, time


def main():
    data_path = '/home/grainger/Desktop/datasets/LTMMD/long-term-movement-monitoring-database-1.0.0/CO001'
    output_path = '/home/grainger/Desktop/skdh_testing/ltmm/'
    ltmm_skdh = LTMM_SKDH()
    ltmm_skdh.run_ltmm_skdh(data_path, output_path)


if __name__ == '__main__':
    main()


