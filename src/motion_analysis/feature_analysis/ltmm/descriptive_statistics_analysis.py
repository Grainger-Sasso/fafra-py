import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.datasets.ltmm.ltmm_risk_assessment import LTMMRiskAssessment
from src.datasets.ltmm.ltmm_dataset import LTMMData
from src.motion_analysis.filters.motion_filters import MotionFilters


class DescriptiveStatisticsAnalysis:
    def __init__(self):
        self.motion_filters = MotionFilters()

    def analyze_descriptive_statistics(self):
        ltmm_dataset_name = 'LabWalks'
        # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
        ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
        clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
        report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
        ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)


        # Filter the data
        ltmm_ra.apply_lp_filter()
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(False)

        faller_mean_std = self._assess_data_mean_std(ltmm_faller_data)
        non_faller_mean_std = self._assess_data_mean_std(ltmm_non_faller_data)

        faller_mean_v = faller_mean_std[0][:, 0]
        faller_std_v = faller_mean_std[1][:, 0]
        faller_rms_v = faller_mean_std[2][:, 0]
        non_faller_mean_v = non_faller_mean_std[0][:, 0]
        non_faller_std_v = non_faller_mean_std[1][:, 0]
        non_faller_rms_v = non_faller_mean_std[2][:, 0]

        faller_mean_ml = faller_mean_std[0][:, 1]
        faller_std_ml = faller_mean_std[1][:, 1]
        faller_rms_ml = faller_mean_std[2][:, 1]
        non_faller_mean_ml = non_faller_mean_std[0][:, 1]
        non_faller_std_ml = non_faller_mean_std[1][:, 1]
        non_faller_rms_ml = non_faller_mean_std[2][:, 1]

        faller_mean_ap = faller_mean_std[0][:, 2]
        faller_std_ap = faller_mean_std[1][:, 2]
        faller_rms_ap = faller_mean_std[2][:, 2]
        non_faller_mean_ap = non_faller_mean_std[0][:, 2]
        non_faller_std_ap = non_faller_mean_std[1][:, 2]
        non_faller_rms_ap = non_faller_mean_std[2][:, 2]

        fig7, ax7 = plt.subplots()
        ax7.set_title('Vertical Axis (faller/non - mean, std, rms)')
        data = [faller_mean_v, non_faller_mean_v, faller_std_v, non_faller_std_v, faller_rms_v, non_faller_rms_v]
        ax7.boxplot(data)

        fig8, ax8 = plt.subplots()
        ax8.set_title('ML Axis (faller/non - mean, std, rms)')
        data = [faller_mean_ml, non_faller_mean_ml, faller_std_ml, non_faller_std_ml, faller_rms_ml, non_faller_rms_ml]
        ax8.boxplot(data)

        fig9, ax9 = plt.subplots()
        ax9.set_title('AP Axis (faller/non - mean, std, rms)')
        data = [faller_mean_ap, non_faller_mean_ap, faller_std_ap, non_faller_std_ap, faller_rms_ap, non_faller_rms_ap]
        ax9.boxplot(data)

        plt.xticks([1, 2, 3, 4, 5, 6],
                   ['Faller Mean', 'Non-faller Mean', 'Faller Std. Dev.', 'Non-faller Std. Dev.', 'Faller RMS',
                    'Non-faller RMS'])

        plt.show()


    def _assess_data_mean_std(self, ltmm_data: List[LTMMData]):
        agg_mean = []
        agg_std = []
        agg_rms = []
        for data in ltmm_data:
            data = np.array(data.get_data())
            data = data[:, 0:3]
            agg_mean.append(data.mean(axis=0))
            agg_std.append(data.std(axis=0))
            agg_rms.append([self.motion_filters.calculate_rms(data[:, 0]), self.motion_filters.calculate_rms(data[:,1]), self.motion_filters.calculate_rms(data[:, 2])])
        return np.array(agg_mean), np.array(agg_std), np.array(agg_rms)


def main():
    dsa = DescriptiveStatisticsAnalysis()
    dsa.analyze_descriptive_statistics()

if __name__ == '__main__':
    main()
