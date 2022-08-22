import datetime
import csv
import numpy as np

from src.fibion_mvp.skdh_pipeline import SKDHPipelineGenerator
from src.fibion_mvp.skdh_pipeline import SKDHPipelineRunner

def run_pipeline(path, output_path):
    gait_metric_names = [
        'PARAM:gait speed',
        'BOUTPARAM:gait symmetry index',
        'PARAM:cadence',
        'Bout Steps',
        'Bout Duration',
        'Bout N',
        'Bout Starts',
        # Additional gait params
        'PARAM:stride time',
        'PARAM:stride time asymmetry',
        'PARAM:stance time',
        'PARAM:stance time asymmetry',
        'PARAM:swing time',
        'PARAM:swing time asymmetry',
        'PARAM:step time',
        'PARAM:step time asymmetry',
        'PARAM:initial double support',
        'PARAM:initial double support asymmetry',
        'PARAM:terminal double support',
        'PARAM:terminal double support asymmetry',
        'PARAM:double support',
        'PARAM:double support asymmetry',
        'PARAM:single support',
        'PARAM:single support asymmetry',
        'PARAM:step length',
        'PARAM:step length asymmetry',
        'PARAM:stride length',
        'PARAM:stride length asymmetry',
        'PARAM:gait speed asymmetry',
        'PARAM:intra-step covariance - V',
        'PARAM:intra-stride covariance - V',
        'PARAM:harmonic ratio - V',
        'PARAM:stride SPARC',
        'BOUTPARAM:phase coordination index',
        'PARAM:intra-step covariance - V',
        'PARAM:intra-stride covariance - V',
        'PARAM:harmonic ratio - V',
        'PARAM:stride SPARC',
        'BOUTPARAM:phase coordination index'
    ]
    # read data into numpy array
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter= ',')
        row_num = 0
        header = None
        x_data = []
        y_data = []
        z_data = []
        time = []
        for row in reader:
            x_data.append(float(row['x-axis (g)']))
            y_data.append(float(row['y-axis (g)']))
            z_data.append(float(row['z-axis (g)']))
            time.append(float(row['epoc (ms)']) / 1000.0)
        f.close()
    x_data = x_data[2383971:]
    y_data = y_data[2383971:]
    z_data = z_data[2383971:]
    time = time[2383971:]
    data = np.array([x_data, y_data, z_data])
    data = data.T
    time = np.array(time)
    day_ends = np.array([[0, 3836477], [3836477, (len(time) - 1)]])
    fs = 100.0
    print(path)
    # push data into gait pipeline
    pipeline_gen = SKDHPipelineGenerator()
    # gait_pipeline = pipeline_gen.generate_gait_pipeline(output_path)
    # gait_pipeline_run = SKDHPipelineRunner(gait_pipeline, gait_metric_names)
    # results = gait_pipeline_run.run_gait_pipeline(data, time, fs, day_ends)
    pipeline = pipeline_gen.generate_pipeline(output_path)
    pipeline_run = SKDHPipelineRunner(pipeline, gait_metric_names)
    results = pipeline_run.run_pipeline(data, time, fs, day_ends)
    for name, result in results.items():
        print(name)
        print(result)
        for nest_name, nest_result in result.items():
            print(nest_name)
            print(nest_result)
    # check results
    pass

def get_day_ends(time):
    current_ix = 0
    iter_ix = 0
    day_end_pairs = []
    while iter_ix + 1 <= len(time) - 1:
        if datetime.fromtimestamp(time[iter_ix]).time().hour > datetime.fromtimestamp(
                time[iter_ix + 1]).time().hour:
            day_end_pairs.append([current_ix, iter_ix])
            current_ix = iter_ix
        iter_ix += 1
    day_end_pairs.append([current_ix, len(time) - 1])
    return day_end_pairs

def main():
    path = '/home/grainger/Desktop/datasets/mbientlab/test/MULTIDAY_MetaWear_2022-08-19T12.38.00.909_C85D72EF7FA2_Accelerometer.csv'
    output_path = '/home/grainger/Desktop/datasets/mbientlab/output/'
    run_pipeline(path, output_path)


if __name__ == '__main__':
    main()
