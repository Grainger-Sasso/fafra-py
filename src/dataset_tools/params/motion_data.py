import pandas as pd
from typing import List
from src.dataset_tools.motion_data import TriaxialLinearAcceleration
from src.dataset_tools.motion_data import TriaxialAngularAcceleration
from src import Subject
from src import Activity


class MotionData:

    def __init__(self, subject: Subject, activity: Activity, trial: str,
                 motion_df: pd.DataFrame, tri_lin_accs: List[TriaxialLinearAcceleration],
                 tri_ang_accs: List[TriaxialAngularAcceleration]):
        self.subject: Subject = subject
        self.activity: Activity = activity
        self.trial: str = trial
        self.motion_df: pd.DataFrame = motion_df
        self.tri_lin_accs: List[TriaxialLinearAcceleration] = tri_lin_accs
        self.tri_ang_accs: List[TriaxialAngularAcceleration] = tri_ang_accs

    def get_subject(self):
        return self.subject

    def get_activity(self):
        return self.activity

    def get_trial(self):
        return self.trial

    def get_motion_df(self):
        return self.motion_df

    def get_tri_lin_accs(self) -> List[TriaxialLinearAcceleration]:
        return self.tri_lin_accs

    def get_tri_ang_accs(self) -> List[TriaxialAngularAcceleration]:
        return self.tri_ang_accs

    def get_triaxial_accs(self):
        # Flattens list of lists
        accs = [self.tri_lin_accs, self.tri_ang_accs]
        return [acc for sublist in accs for acc in sublist]

    def calculate_first_derivative_data(self):
        for tri_acc in self.get_triaxial_accs():
            for acc in tri_acc.get_all_axes():
                acc.set_lp_filtered_data(self.filters.apply_lpass_filter(acc.acceleration_data, self.sampling_rate))

