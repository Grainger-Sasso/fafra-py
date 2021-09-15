import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


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
        data = pd.DataFrame()
        ax = sns.violinplot(x="metric", y="scaled_value", hue="faller",
                            data=tips, palette="muted")

    def _generate_metric_plot_labels(self, metric_names):
        labels = []
        labels.extend(metric_names)
        pass

    def _separate_faller_nonfaller_data(self, x, y):
        data = []
        for metric in x.T:
            faller = [val for ix, val in enumerate(metric) if y[ix] == 1]
            non_faller = [val for ix, val in enumerate(metric) if y[ix] == 0]
            data.extend([faller, non_faller])
        return data