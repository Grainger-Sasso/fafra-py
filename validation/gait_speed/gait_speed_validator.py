import numpy as np

from src.risk_assessments.fall_risk_assessment import FallRiskAssessment
from src.motion_analysis.gait_analysis.gait_analyzer import GaitAnalyzer
from src.dataset_tools.dataset_builders.builder_instances.uiuc_gait_dataset_builder import DatasetBuilder
from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.dataset_tools.risk_assessment_data.user_data import UserData
from src.dataset_tools.risk_assessment_data.imu_data import IMUData


class GaitSpeedValidator:
    def __init__(self):
        """
        Constructor of the GaitSpeedValidator class
        For the UIUC gait speed data, see for truth data:
        root folder -> Fixed speed data_Instrumented Treadmill
        """
        self.subj_gs_truth = {
            '101 ': {'CWT': 1.25, 'BS': 1.25},
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

    def validate_gait_speed_estimator(self, dataset: Dataset):
        # Instantiate gait analyzer and run the dataset through the gait analyzer
        ga = GaitAnalyzer()
        # Compare the results of the gait analyzer with truth values
        gs_results = [{
            'id': user_data.get_clinical_demo_data().get_id(),
            'gait_speed': ga.estimate_gait_speed(user_data)
                       } for user_data in dataset.get_dataset()]
        cwt_diffs = []
        bs_diffs = []
        for result in gs_results:
            cwt_truth_value = self.subj_gs_truth[result['id']]['CWT']
            bs_truth_value = self.subj_gs_truth[result['id']]['BS']
            cwt_diffs.append(abs(cwt_truth_value - result['gait_speed']))
            bs_diffs.append(abs(bs_truth_value - result['gait_speed']))
        cwt_diffs = np.array(cwt_diffs)
        bs_diffs = np.array(bs_diffs)
        return cwt_diffs, bs_diffs



def main():
    # Instantiate the Validator
    val = GaitSpeedValidator()
    # Set dataset paths and builder parameters
    dataset_path = r'C:\Users\gsass\Documents\fafra\datasets\GaitSpeedValidation\GaitSpeedValidation\Hexoskin Binary Data files 2\Hexoskin Binary Data files'
    clinical_demo_path = 'N/A'
    segment_dataset = False
    epoch_size = 0.0
    # Instantiate the builder and build the dataset
    db = DatasetBuilder()
    dataset = db.build_dataset(dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size)
    # Run dataset through low-pass filter
    fra = FallRiskAssessment(LightGBMRiskClassifier({}))
    for user_data in dataset:
        fra.apply_lp_filter(user_data)
    # Run the validation
    cwt_diffs, bs_diffs = val.validate_gait_speed_estimator(dataset)
    print(min(cwt_diffs))
    print(max(cwt_diffs))
    print(cwt_diffs.mean())
    print('\n\n\n')
    print(min(bs_diffs))
    print(max(bs_diffs))
    print(bs_diffs.mean())


if __name__ == '__main__':
    main()
