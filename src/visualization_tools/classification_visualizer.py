import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ClassificationVisualizer:

    def plot_data(self, x, y):
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter')
        plt.show()

    def plot_classification(self, model, x, y):
        ax = plt.gca()
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='winter')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        grid_x = np.linspace(xlim[0], xlim[1], 30)
        grid_y = np.linspace(ylim[0], ylim[1], 30)
        grid_Y, grid_X = np.meshgrid(grid_y, grid_x)
        xy = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T
        P = model.decision_function(xy).reshape(grid_X.shape)

        # plot decision boundary and margins
        ax.contour(grid_X, grid_Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # plot support vectors

        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.show()



