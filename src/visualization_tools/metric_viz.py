import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.risk_classification.input_metrics.metric_names import MetricNames


class MetricViz:

    def __init__(self):
        pass

    def boxplot_metrics(self, x, y, metric_names):
        data = self._separate_faller_nonfaller_data(x, y)
        fig, ax = plt.subplots()
        ax.boxplot(data)
        plt.show()

    def violin_plot_metrics(self, x, y, metric_names):
        data = self._separate_faller_nonfaller_data(x, y)
        metric_plot_labels = self._generate_metric_plot_labels(metric_names)
        pd_data = []
        for label, metric in zip(metric_plot_labels, data):
            if 'nonfaller' in label:
                fs = 'nonfaller'
            else:
                fs = 'faller'
            for i in metric:
                pd_data.append([label, i, fs])
        data = pd.DataFrame(data=pd_data,
                            columns=['labels', 'metrics', 'fallers'])
        ax = sns.violinplot(x="labels", y="metrics", hue="fallers",
                            data=data, palette='muted')
        plt.show()

    def _generate_metric_plot_labels(self, metric_names):
        labels = []
        for name in metric_names:
            if name == MetricNames.AUTOCORRELATION:
                labels.append('ac_x_faller')
                labels.append('ac_x_nonfaller')
                labels.append('ac_y_faller')
                labels.append('ac_y_nonfaller')
            elif name == MetricNames.FAST_FOURIER_TRANSFORM:
                labels.append('fft_x_faller')
                labels.append('fft_x_nonfaller')
                labels.append('fft_y_faller')
                labels.append('fft_y_nonfaller')
            else:
                labels.append(name.get_value() + '_faller')
                labels.append(name.get_value() + '_nonfaller')
        return labels

    def _separate_faller_nonfaller_data(self, x, y):
        data = []
        for metric in x.T:
            faller = [val for ix, val in enumerate(metric) if y[ix] == 1]
            non_faller = [val for ix, val in enumerate(metric) if y[ix] == 0]
            data.extend([faller, non_faller])
        return data