import seaborn as sns
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import gridspec

from src.risk_classification.input_metrics.metric_names import MetricNames


class MetricViz:

    def __init__(self):
        pass

    def boxplot_metrics(self, x, y, metric_names):
        data = self._separate_faller_nonfaller_data(x, y)
        fig, ax = plt.subplots()
        ax.boxplot(data)
        plt.show()

    def violin_plot_metrics(self, x, y):
        # fig, axes = plt.subplots(1, len(x))
        fig = plt.figure()
        ix = 0
        cols = 3
        rows = int(math.ceil(len(x) / cols))
        gs = gridspec.GridSpec(rows, cols)
        for name, metric in x.items():
            labels = []
            metric_value = metric.get_value()
            faller = np.array([val for ix, val in enumerate(metric_value) if y[ix] == 1])
            non_faller = np.array([val for ix, val in enumerate(metric_value) if y[ix] == 0])
            pd_data = []
            for i in faller:
                pd_data.append({'fall_status': 'faller',
                                'metric': i, 'name': name.get_value()+'_faller'})
            for i in non_faller:
                pd_data.append({'fall_status': 'non_faller',
                                'metric': i, 'name': name.get_value()+'_nonfaller'})
            df = pd.DataFrame(pd_data)
            ax = fig.add_subplot(gs[ix])
            sns.violinplot(x='name', y='metric', hue='fall_status',
                           data=df, ax=ax)
            labels.extend([name.get_value()+'_faller', name.get_value()+'_nonfaller'])
            ix += 1
        fig.tight_layout()
        plt.show()

    def set_axis_style(self, ax, labels):
        ax.set_title('Faller and Nonfaller Metric Distribution', fontsize=10)
        ax.set_ylabel('Observed values', fontsize=10)
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Metric')
        ax.xaxis.label.set_size(10)

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
        labels = []
        fs = []
        for name, metric in x.items():
            faller = [val for ix, val in enumerate(metric) if y[ix] == 1]
            non_faller = [val for ix, val in enumerate(metric) if y[ix] == 0]
            data.extend([faller, non_faller])
            labels.extend([name.get_value(), name.get_value()])
            fs.extend(['faller', 'nonfaller'])
        return data, labels, fs