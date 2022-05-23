import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math
from pandas.plotting import parallel_coordinates
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from matplotlib import gridspec

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.metric_names import MetricNames
 

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

    def parallel_coordinates_plot(self, x, y):
        '''this function plots parallel coordinate plot'''
        
        d={'ac1': x[0],'ac2':x[1],'cov':x[2],'ff1':x[3],'ff2':x[4],'gse':x[5],'mean':x[6],'rms':x[7],'sig':x[8],'sma':x[9],'std':x[10],'zero_c':x[11], 'y_values': y[0]}
        df=pd.DataFrame(d)
        plt.figure(figsize=(2,15))
        parallel_coordinates(df,'y_values',color=('#556270', '#C7F464'))
        plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Features values', fontsize=15)
        plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
        plt.show()

    def all_feature_scatterplot_double(self, x, y):
        '''
        This function plots dataplot from all combinations of feature and label all classification result in graph, but accept x and y data seperately 
        ac1	ac2	cov	ff1	ff2	gse	mean	rms	signal_energy	sma	std	zero_cross	y
        x and y should be dataframe
        '''
        d={'ac1': x[0],'ac2':x[1],'cov':x[2],'ff1':x[3],'ff2':x[4],'gse':x[5],'mean':x[6],'rms':x[7],'sig':x[8],'sma':x[9],'std':x[10],'zero_c':x[11], 'y_values': y[0]}
        df=pd.DataFrame(d)
        plt.figure()
        sns.pairplot(df, hue = "y_values", size=2, markers=["s", "D"])
        plt.show()

    def all_feature_scatterplot(self, all):
        '''
        This function plots dataplot from all combinations of feature and label all classification result in graph  
        '''
        #df=pd.DataFrame({'x_values': x, 'y_values': y})
        plt.figure()
        sns.pairplot(all, hue = "y", size=2, markers=["s", "D"])
        plt.show()

    def plot_decision_boundry(model, x, y):
        value=1.5
        width=0.75
        #X_train2 = pca.fit_transform(x_train)
        cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])
        plot_decision_regions(x,y.astype(int),clf=model,legend=2,feature_index=[0,2],   #column 0 and column 2 one will be plotted  
                    filler_feature_values={1: value, 3:value, 4:value, 5:value, 6:value},  #these will be ignored
                    filler_feature_ranges={1: width, 3: width, 4:width, 5:width, 6:width},X_highlight=x)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x[:,0],x[:,2],c=y,cmap=cmap_bold, edgecolor='k', s=20)
        plt.title('Knn with K='+ str(10))
        plt.show()

    "return distance matrix between clustered features"
    def corr_linkage(self,input_metric:InputMetrics): 
        X, names = input_metric.get_metric_matrix()
        #feature_names=['ac1','ac2','cov','ff1','ff2','gse','mean','rms','sig','sma','std','zero_c']
        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)#calculated distance btw clusters based on incremental algorithm

        dendro = hierarchy.dendrogram(corr_linkage, labels=names, leaf_rotation=90)
        dendro_idx = np.arange(0, len(dendro['ivl']))

        fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        ax2.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        plt.show()

        cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        print(selected_features)
        return corr_linkage

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

    def boxplot_metrics(self, x, y, metric_names):
        data = self._separate_faller_nonfaller_data(x, y)
        fig, ax = plt.subplots()
        ax.boxplot(data)
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

def main():
    rootPath = r'D:\carapace\metric_test_data'
    
    train_x_path = os.path.join(rootPath, '2021_09_13_x_metrics.csv')
    train_y_path = os.path.join(rootPath, '2021_09_13_y_labels.csv')
    data_path=os.path.join(rootPath,'all_data_metrics.csv')
    #all_data = pd.read_csv(data_path)

    train_x = pd.read_csv(train_x_path,header=None)
    
    train_y = pd.read_csv(train_y_path,header=None)

    cl=ClassificationVisualizer()
    cl.all_feature_scatterplot_double(train_x,train_y)
    #cl.all_feature_scatterplot(all_data)



if __name__ ==  '__main__':
    main() 

