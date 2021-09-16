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

    def violin_plot_metrics(self, x, y):
        data, labels, fs = self._separate_faller_nonfaller_data(x, y)
        pd_data = []
        for val, label, fall_s in zip(data, labels, fs):
            for i in val:
                pd_data.append({'fall_status': fall_s,
                                'metric': i, 'name': label})

        df = pd.DataFrame(pd_data)
        faller_df = df.loc[df['fall_status'] == 'faller']
        nonfaller_df = df.loc[df['fall_status'] == 'non_faller']
        grid = sns.FacetGrid(df, col="name")
        for ix, label in enumerate(sorted(set(labels))):
            sns.violinplot(x=df.loc[df['name'] == label]['name'], y='metric', hue="fall_status",
                           data=df.loc[df['name'] == label], ax=grid.axes[0, ix])
        plt.show()
        grid.map(sns.violinplot, x="name", y="metric", hue="fall_status",
                            data=faller_df)
        grid.map(sns.violinplot, x="name", y="metric", hue="fall_status",
                 data=nonfaller_df)

        for label, metric in zip(labels, data):
            if 'nonfaller' in label:
                fs = 'nonfaller'
            else:
                fs = 'faller'
            for i in metric:
                pd_data.append([label, i, fs])
        data = pd.DataFrame(data=pd_data,
                            columns=['labels', 'metrics', 'fallers'])
        for label in labels:
            label = label.split('_')[0]
            plot_data_f = data.loc[data["labels"] == label + '_faller']
            plot_data_nf = data.loc[data["labels"] == label + '_nonfaller']
            plot_data = pd.concat([plot_data_f,plot_data_nf])
            ax = sns.violinplot(x="labels", y="metrics", hue="fallers",
                            data=plot_data)
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
        labels = []
        fs = []
        for name, metric in x.items():
            faller = [val for ix, val in enumerate(metric) if y[ix] == 1]
            non_faller = [val for ix, val in enumerate(metric) if y[ix] == 0]
            data.extend([faller, non_faller])
            labels.extend([name.get_value(), name.get_value()])
            fs.extend(['faller', 'nonfaller'])
        return data, labels, fs