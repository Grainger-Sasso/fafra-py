import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.datasets.ltmm.ltmm_risk_assessment import LTMMRiskAssessment
from src.datasets.ltmm.ltmm_dataset import LTMMData
from src.motion_analysis.filters.motion_filters import MotionFilters
from src.motion_analysis.feature_extraction.frequency_analysis.fast_fourier_transform import FastFourierTransform
from src.motion_analysis.feature_extraction.frequency_analysis.auto_correlation import AutoCorrelation
from src.motion_analysis.peak_detection.peak_detector import PeakDetector


class FrequencyAnalysis:
    def __init__(self):
        self.motion_filters = MotionFilters()
        ltmm_dataset_name = 'LabWalks'
        # ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0'
        ltmm_dataset_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\LabWalks'
        clinical_demo_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ClinicalDemogData_COFL.xlsx'
        report_home_75h_path = r'C:\Users\gsass\Desktop\Fall Project Master\datasets\LTMMD\long-term-movement-monitoring-database-1.0.0\ReportHome75h.xlsx'
        self.ltmm_ra = LTMMRiskAssessment(ltmm_dataset_name, ltmm_dataset_path, clinical_demo_path, report_home_75h_path)


    def apply_lpf(self):
        # Filter the data
        self.ltmm_ra.apply_lp_filter()

    def analyze_fft(self):
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Apply fft to the data
        x_faller_fft, y_faller_fft = self._apply_fft_to_data(ltmm_faller_data)
        x_non_faller_fft, y_non_faller_fft = self._apply_fft_to_data(ltmm_non_faller_data)
        # Plot the results
        # self._plot_fft_overlayed(x_faller_fft, y_faller_fft, 'Faller FFT Overlayed')
        # self._plot_fft_overlayed(x_non_faller_fft, y_non_faller_fft, 'Non-faller FFT Overlayed')
        self._plot_faller_nonfaller_overlayed(x_faller_fft, y_faller_fft, x_non_faller_fft, y_non_faller_fft, 'FFT')
        plt.show()

    def analyze_autocorr(self):
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Apply autocorrelation to data
        x_faller_ac, y_faller_ac = self._apply_autocorr_to_data(ltmm_faller_data)
        x_non_faller_acc, y_non_faller_ac = self._apply_autocorr_to_data(ltmm_non_faller_data)
        # Get first peak loactions of ac data
        faller_peak_locs = self._get_peak_locs(x_faller_ac, y_faller_ac)
        non_faller_peak_locs = self._get_peak_locs(x_non_faller_acc, y_non_faller_ac)
        # Plot distribution of first peak ixs
        data = [faller_peak_locs, non_faller_peak_locs]
        plt.boxplot(data)
        plt.xticks([1, 2], ['faller_peak_locs', 'non_faller_peak_locs'])
        # self._plot_faller_nonfaller_overlayed(x_faller_ac, y_faller_ac, x_non_faller_acc, y_non_faller_ac, 'Autocorrelation')
        plt.show()


    def _get_peak_locs(self, x_dataset,y_dataset):
        peak_detector = PeakDetector()
        peak_locs = []
        for x_data, y_data in zip(x_dataset, y_dataset):
            peak_ixs = peak_detector.detect_peaks(y_data)
            if len(peak_ixs) > 0:
                peak1 = peak_ixs[0]
                peak_loc = x_data[peak1]
                peak_locs.append(peak_loc)
            else:
                continue
        return peak_locs

    def _apply_autocorr_to_data(self, ltmm_dataset):
        x_dataset_ac = []
        y_dataset_ac = []
        ac = AutoCorrelation()
        for ltmm_data in ltmm_dataset:
            data = ltmm_data.get_data()
            v_axis_acc_data = data.T[0]
            x_ac, y_ac = ac.autocorrelate(v_axis_acc_data)
            x_dataset_ac.append(x_ac)
            y_dataset_ac.append(y_ac)
        return x_dataset_ac, y_dataset_ac

    def _apply_fft_to_data(self, ltmm_dataset):
        x_dataset_fft = []
        y_dataset_fft = []
        fft = FastFourierTransform()
        for ltmm_data in ltmm_dataset:
            sampling_rate = ltmm_data.get_sampling_frequency()
            data = ltmm_data.get_data()
            v_axis_acc_data = data.T[0]
            x_fft, y_fft = fft.perform_fft(v_axis_acc_data, sampling_rate)
            x_dataset_fft.append(x_fft)
            y_dataset_fft.append(y_fft)
        return x_dataset_fft, y_dataset_fft

    def _plot_fft_overlayed(self, x_fft, y_fft, title):
        fig, ax = plt.subplots()
        for x, y in zip(x_fft, y_fft):
            ax.plot(x, y, color='black', alpha=0.4)
        plt.title(title)

    def _plot_faller_nonfaller_overlayed(self, x_faller, y_faller, x_nonfaller, y_nonfaller, type):
        for x, y in zip(x_faller, y_faller):
            plt.plot(x, y, color='red', alpha=0.2)
        for x, y, in zip(x_nonfaller, y_nonfaller):
            plt.plot(x, y, color='blue', alpha=0.2)
        plt.title(f'Faller (red) and Non-faller (blue) {type} overlayed')


    # def analyze_descriptive_statistics(self):
    #
    #     faller_mean_std = self._assess_data_mean_std(ltmm_faller_data)
    #     non_faller_mean_std = self._assess_data_mean_std(ltmm_non_faller_data)
    #
    #     faller_mean_v = faller_mean_std[0][:, 0]
    #     faller_std_v = faller_mean_std[1][:, 0]
    #     faller_rms_v = faller_mean_std[2][:, 0]
    #     non_faller_mean_v = non_faller_mean_std[0][:, 0]
    #     non_faller_std_v = non_faller_mean_std[1][:, 0]
    #     non_faller_rms_v = non_faller_mean_std[2][:, 0]
    #
    #     faller_mean_ml = faller_mean_std[0][:, 1]
    #     faller_std_ml = faller_mean_std[1][:, 1]
    #     faller_rms_ml = faller_mean_std[2][:, 1]
    #     non_faller_mean_ml = non_faller_mean_std[0][:, 1]
    #     non_faller_std_ml = non_faller_mean_std[1][:, 1]
    #     non_faller_rms_ml = non_faller_mean_std[2][:, 1]
    #
    #     faller_mean_ap = faller_mean_std[0][:, 2]
    #     faller_std_ap = faller_mean_std[1][:, 2]
    #     faller_rms_ap = faller_mean_std[2][:, 2]
    #     non_faller_mean_ap = non_faller_mean_std[0][:, 2]
    #     non_faller_std_ap = non_faller_mean_std[1][:, 2]
    #     non_faller_rms_ap = non_faller_mean_std[2][:, 2]
    #
    #     fig7, ax7 = plt.subplots()
    #     ax7.set_title('Vertical Axis (faller/non - mean, std, rms)')
    #     data = [faller_mean_v, non_faller_mean_v, faller_std_v, non_faller_std_v, faller_rms_v, non_faller_rms_v]
    #     ax7.boxplot(data)
    #
    #     fig8, ax8 = plt.subplots()
    #     ax8.set_title('ML Axis (faller/non - mean, std, rms)')
    #     data = [faller_mean_ml, non_faller_mean_ml, faller_std_ml, non_faller_std_ml, faller_rms_ml,
    #             non_faller_rms_ml]
    #     ax8.boxplot(data)
    #
    #     fig9, ax9 = plt.subplots()
    #     ax9.set_title('AP Axis (faller/non - mean, std, rms)')
    #     data = [faller_mean_ap, non_faller_mean_ap, faller_std_ap, non_faller_std_ap, faller_rms_ap,
    #             non_faller_rms_ap]
    #     ax9.boxplot(data)
    #
    #     plt.xticks([1, 2, 3, 4, 5, 6],
    #                ['Faller Mean', 'Non-faller Mean', 'Faller Std. Dev.', 'Non-faller Std. Dev.', 'Faller RMS',
    #                 'Non-faller RMS'])
    #
    #     plt.show()


def main():
    fa = FrequencyAnalysis()
    # fa.analyze_fft()
    fa.analyze_autocorr()

if __name__ == '__main__':
    main()
