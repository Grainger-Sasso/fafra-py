from matplotlib import pyplot as plt

class MetricViz:

    def __init__(self):
        pass

    def boxplot_metrics(self, x, y):
        data = []
        for metric in x.T:
            faller = [val for ix, val in enumerate(metric) if y[ix] == 1]
            non_faller = [val for ix, val in enumerate(metric) if y[ix] == 0]
            data.extend([faller, non_faller])
        fig, ax = plt.subplots()
        ax.boxplot(data)
        plt.show()
