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
        for row in reader:
            x_data.append(float(row['X']))
            y_data.append(float(row['Y']))
            z_data.append(float(row['Z']))
    data = np.array([x_data, y_data, z_data])
    data = data.T
    fs = 100.0
    time = np.linspace(0, (len(x_data) - 1) / fs, len(x_data))
    print(path)
    # push data into gait pipeline
    pipeline_gen = SKDHPipelineGenerator()
    gait_pipeline = pipeline_gen.generate_gait_pipeline(output_path)
    gait_pipeline_run = SKDHPipelineRunner(gait_pipeline, gait_metric_names)
    results = gait_pipeline_run.run_gait_pipeline(data, time, fs, [[]])
    print(results)
    # check results
    pass


def main():
    path = '/home/grainger/Desktop/datasets/mbientlab/EA78C3D3F08A_MetaWear_acceleration_2022-08-17T08.06.49.448.csv'
    output_path = '/home/grainger/Desktop/datasets/mbientlab/output/'
    run_pipeline(path, output_path)


if __name__ == '__main__':
    main()
