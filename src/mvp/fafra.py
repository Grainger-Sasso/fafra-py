import os
import glob
import json
from typing import List


from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.mvp.report_generation.report_generator import ReportGenerator
from src.mvp.mbientlab_dataset_builder import MbientlabDatasetBuilder


class FaFRA:
    def __init__(self):
        self.custom_metric_names = tuple(
            [
                MetricNames.SIGNAL_MAGNITUDE_AREA,
                MetricNames.COEFFICIENT_OF_VARIANCE,
                MetricNames.STANDARD_DEVIATION,
                MetricNames.MEAN,
                MetricNames.SIGNAL_ENERGY,
                MetricNames.ROOT_MEAN_SQUARE
            ]
        )
        self.gait_metric_names: List[str] = [
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
        self.ra_model_path = ''

    def perform_risk_assessment(self, assessment_path):
        # Generate risk metrics
        ra_metrics = MetricGen().generate_ra_metrics(assessment_path, self.custom_metric_names, self.gait_metric_names)
        # Assess risk using risk model
        ra_results = Model().assess_fall_risk(self.ra_model_path)
        # Generate risk report
        rg = ReportGenerator()
        rg.generate_report(assessment_path, '', '', '')


class MetricGen:
    def __init__(self):
        pass

    def generate_ra_metrics(self, assessment_path, custom_metric_names, gait_metric_names):
        data = DataLoader().load_data(assessment_path)
        print('yes')


class DataLoader:
    def load_json_data(self, file_path):
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_data(self, assessment_path):
        # Load in the assessment data file (contains device type of the IMU data)
        assess_info_path = os.path.join(assessment_path, 'assessment_info.json')
        assess_info = self.load_json_data(assess_info_path)
        data_type = assess_info['device_type']
        # Load in the user data json file (contains demographic data of the user)
        collect_data_path = os.path.join(assessment_path, 'collected_data')
        user_info_path = os.path.join(collect_data_path, 'user_info.json')
        user_info = self.load_json_data(user_info_path)
        # Load in the IMU data file based on type
        imu_data_filename = [filename for filename in os.listdir(collect_data_path) if filename.startswith("imu_data")][0]
        imu_data_path = os.path.join(collect_data_path, imu_data_filename)
        # Build dataset objects
        return self.build_dataset('mbientlab', imu_data_path, user_info)

    def build_dataset(self, data_type, imu_data_path, user_info):
        if data_type.lower() == 'mbientlab':
            user_data: List[UserData] = MbientlabDatasetBuilder().build_single_user(imu_data_path, user_info)
            # TODO: may need to define ra data objects specific to the MVP
        else:
            raise ValueError(f'Unknown IMU datatype provided {data_type}')
        dataset = Dataset(
            'mbientlab', [imu_data_path], [], user_data, {}
        )
        return dataset


class Model:
    def __init__(self):
        pass

    def assess_fall_risk(self, model_path):
        pass


def main():
    fafra = FaFRA()
    path = '/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/'
    ra = fafra.perform_risk_assessment(path)


if __name__ == '__main__':
    main()
