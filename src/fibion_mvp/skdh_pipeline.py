import os

import numpy as np
from skdh import Pipeline
from skdh.preprocessing import DetectWear
from skdh.gait import Gait
from skdh.sleep import Sleep
from skdh.activity import ActivityLevelClassification


class SKDHPipelineGenerator:
    def generate_pipeline(self, output_path):
        pipeline = Pipeline()
        # pipeline.add(DetectWear())
        gait_result_file = os.path.join(output_path, 'gait_results.csv')
        # gait_plot_file = os.path.join(output_path, 'gait_plot.pdf')
        # pipeline.add(Gait(), save_file=gait_result_file, plot_file=gait_plot_file)
        pipeline.add(Gait(), save_file=gait_result_file)
        pipeline.add(Gait())
        act_result_file = os.path.join(output_path, 'activity_results.csv')
        act_plot_file = os.path.join(output_path, 'activity_plot.pdf')
        act = ActivityLevelClassification(cutpoints='esliger_lumbar_adult')
        # act.add(TotalIntensityTime(level='sed', epoch_length=60, cutpoints='esliger_lumbar_adult'))
        # pipeline.add(act, save_file=act_result_file, plot_file=act_plot_file)
        pipeline.add(act, save_file=act_result_file)
        sleep_result_file = os.path.join(output_path, 'sleep_results.csv')
        # sleep_plot_file = os.path.join(output_path, 'sleep_plot.pdf')
        # pipeline.add(Sleep(day_window=(12, 24)), save_file=sleep_result_file, plot_file=sleep_plot_file)
        pipeline.add(Sleep(day_window=(12, 24)), save_file=sleep_result_file)
        return pipeline

    def generate_gait_pipeline(self, output_path):
        pipeline = Pipeline()
        # gait_result_file = os.path.join(output_path, 'segmented_gait_results.csv')
        # pipeline.add(Gait(), save_file=gait_result_file)
        pipeline.add(Gait())
        return pipeline


class SKDHPipelineRunner:
    def __init__(self, pipeline: Pipeline, gait_metric_names):
        self.pipeline: Pipeline = pipeline
        self.gait_metric_names = gait_metric_names
        self.act_metric_names = [
            "N hours",
            "N wear hours",
            "N wear wake hours",
            'wake sed 5s epoch [min]',
            'wake light 5s epoch [min]',
            'wake mod 5s epoch [min]',
            'wake vig 5s epoch [min]',
            'wake sed avg duration',
        ]
        self.sleep_metric_names = [
            "Day N",
            "Date",
            "TSO Start Timestamp",
            "TSO Start",
            "TSO Duration",
            'average sleep duration',
            'total sleep time',
            'percent time asleep',
            'number of wake bouts',
            'sleep average hazard',
        ]
        self.param_fail_count = 0

    def run_pipeline(self, data, time, fs, day_ends=np.array([])):
        # TODO: list data shape here
        data = np.ascontiguousarray(data)
        # TODO: configure user height
        results = self.pipeline.run(time=time, accel=data, fs=fs, height=1.77, day_ends={(12, 24): day_ends})
        gait_metrics = self.parse_results(results, 'Gait', self.gait_metric_names)
        gait_metrics = self.parse_gait_metrics(gait_metrics)
        act_metrics = self.parse_results(results, 'ActivityLevelClassification', self.act_metric_names)
        sleep_metrics = self.parse_results(results, 'Sleep', self.sleep_metric_names)
        return {'gait_metrics': gait_metrics, 'act_metrics': act_metrics, 'sleep_metrics': sleep_metrics}

    def run_gait_pipeline(self, data, time, fs, day_ends=np.array([])):
        # TODO: list data shape here
        data = np.ascontiguousarray(data)
        # TODO: configure user height
        results = self.pipeline.run(time=time, accel=data, fs=fs, height=1.77, day_ends={(12, 24): day_ends})
        gait_metrics = self.parse_results(results, 'Gait', self.gait_metric_names)
        gait_metrics = self.parse_gait_metrics(gait_metrics)
        return {'gait_metrics': gait_metrics}

    def parse_results(self, results, results_type, metric_names):
        results_act = results[results_type]
        act_metrics = {}
        for name in metric_names:
            if name in results_act.keys():
                act_metrics[name] = results_act[name]
            else:
                act_metrics[name] = np.array([np.nan])
                self.param_fail_count += 1
        return act_metrics

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
        bout_steps, bout_durs, bout_starts = self.parse_bouts(gait_metrics)
        step_sum = np.array(bout_steps).sum()
        duration_sum = np.array(bout_durs).sum()
        new_gait_metrics['Bout Steps: sum'] = float(step_sum)
        new_gait_metrics['Bout Duration: sum'] = float(duration_sum)
        new_gait_metrics['Bout Starts'] = bout_starts
        new_gait_metrics['Bout Duration'] = bout_durs
        return new_gait_metrics

    def parse_bouts(self, gait_metrics):
        bout_steps = []
        bout_durs = []
        bout_starts = []
        bout_n = gait_metrics['Bout N']
        current_ix = 0
        for ix, bout in enumerate(bout_n):
            if bout > current_ix:
                current_ix = bout
                bout_steps.append(gait_metrics['Bout Steps'][ix])
                bout_durs.append(gait_metrics['Bout Duration'][ix])
                bout_starts.append(gait_metrics['Bout Starts'][ix])

        return bout_steps, bout_durs, bout_starts
