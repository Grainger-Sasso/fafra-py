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
        sampling_freq = 100.0
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        self.apply_lpf()
        ltmm_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        # ltmm_faller_data = [data.get_data() for data in ltmm_faller_data]
        ltmm_non_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # ltmm_non_faller_data = [data.get_data() for data in ltmm_non_faller_data]
        epoch_size = 30.00
        for data in ltmm_faller_data:
            data.segment_data(epoch_size)
        for data in ltmm_non_faller_data:
            data.segment_data(epoch_size)
        ltmm_faller_data = [data.get_data_segments() for data in ltmm_faller_data]
        ltmm_faller_data = [item for sublist in ltmm_faller_data for item in sublist]
        ltmm_non_faller_data = [data.get_data_segments() for data in ltmm_non_faller_data]
        ltmm_non_faller_data = [item for sublist in ltmm_non_faller_data for item in sublist]
        # Apply fft to the data
        x_faller_fft, y_faller_fft = self._apply_fft_to_data(ltmm_faller_data, sampling_freq)
        x_non_faller_fft, y_non_faller_fft = self._apply_fft_to_data(ltmm_non_faller_data, sampling_freq)
        # Plot the results
        # self._plot_fft_overlayed(x_faller_fft, y_faller_fft, 'Faller FFT Overlayed')
        # self._plot_fft_overlayed(x_non_faller_fft, y_non_faller_fft, 'Non-faller FFT Overlayed')
        self._plot_faller_nonfaller_overlayed_bars(x_faller_fft, y_faller_fft, x_non_faller_fft, y_non_faller_fft, 'FFT')
        plt.show()

    def analyze_autocorr(self):
        # Separate dataset into fallers and nonfallers, perform rest of steps for each group
        ltmm_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(True)
        ltmm_non_faller_data = self.ltmm_ra.ltmm_dataset.get_ltmm_data_by_faller_status(False)
        # Segment the data by given epoch size
        epoch_size = 9.99
        for data in ltmm_faller_data:
            data.segment_data(epoch_size)
        for data in ltmm_non_faller_data:
            data.segment_data(epoch_size)
        ltmm_faller_data = [data.get_data_segments() for data in ltmm_faller_data]
        ltmm_faller_data = [item for sublist in ltmm_faller_data for item in sublist]
        ltmm_non_faller_data = [data.get_data_segments() for data in ltmm_non_faller_data]
        ltmm_non_faller_data = [item for sublist in ltmm_non_faller_data for item in sublist]
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
            v_axis_acc_data = ltmm_data.T[0]
            x_ac, y_ac = ac.autocorrelate(v_axis_acc_data)
            x_dataset_ac.append(x_ac)
            y_dataset_ac.append(y_ac)
        return x_dataset_ac, y_dataset_ac

    def _apply_fft_to_data(self, ltmm_dataset, sampling_freq):
        x_dataset_fft = []
        y_dataset_fft = []
        fft = FastFourierTransform()
        for data in ltmm_dataset:
            v_axis_acc_data = data.T[0]
            x_fft, y_fft = fft.perform_fft(v_axis_acc_data, sampling_freq)
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
            plt.plot(x, y, color='red', alpha=0.1)
        for x, y, in zip(x_nonfaller, y_nonfaller):
            plt.plot(x, y, color='blue', alpha=0.1)
        plt.title(f'Faller (red) and Non-faller (blue) {type} overlayed')

    def _plot_faller_nonfaller_overlayed_bars(self, x_faller, y_faller, x_nonfaller, y_nonfaller, type):
        plt.plot(x_faller, y_faller, color='red', alpha=0.5)
        plt.plot(x_nonfaller, y_nonfaller, color='blue', alpha=0.5)
        plt.title(f'Faller (red) and Non-faller (blue) {type} overlayed')


def main():
    fa = FrequencyAnalysis()
    fa.analyze_fft()
    # fa.analyze_autocorr()

if __name__ == '__main__':
    main()
