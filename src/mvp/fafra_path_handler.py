import os


class PathHandler:
    def __init__(self, assessment_path):
        # Assessment data paths
        self.assessment_folder = assessment_path
        self.assessment_info_file = ''
        # Collected data paths
        self.collected_data_folder = ''
        self.user_info_file = ''
        self.imu_data_file = ''
        # Generated data paths
        self.generated_data_folder = ''
        # TODO: fill in files for generated data
        self.risk_report_folder = ''
        self.skdh_pipeline_results_folder = ''
        self.ra_model_folder = ''
        self.ra_model_metrics_folder = ''
        self.build_paths()

    def build_paths(self):
        # Assessment data paths
        self.assessment_info_file = os.path.join(self.assessment_folder, 'assessment_info.json')
        # Collected data paths
        self.collected_data_folder = os.path.join(self.assessment_folder, 'collected_data')
        self.user_info_file = os.path.join(self.collected_data_folder, 'user_info.json')
        imu_data_filename = [filename for filename in os.listdir(self.collected_data_folder) if filename.startswith("imu_data")][0]
        self.imu_data_file = os.path.join(self.collected_data_folder, imu_data_filename)
        # Generated data paths
        self.generated_data_folder = os.path.join(self.assessment_folder, 'generated_data')
        self.risk_report_folder = os.path.join(self.generated_data_folder, 'risk_report')
        self.skdh_pipeline_results_folder = os.path.join(self.generated_data_folder, 'skdh_pipeline_results')
        self.ra_model_folder = os.path.join(self.generated_data_folder, 'ra_model')
        self.ra_model_metrics_folder = os.path.join(self.generated_data_folder, 'ra_model_metrics')
