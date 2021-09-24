import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict

from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier


def main():
    start = time.time()
    # Instantiate the classifier
    # classifier = SVMRiskClassifier()
    classifier = KNNRiskClassifier()
    #classifier = LightGBMRiskClassifier({})

    # Read in data/generate data with random seed set
    dg = DataGenerator()
    cv = ClassificationVisualizer()
    #x, y = dg.generate_data()
    rootPath = r'D:\carapace\metric_test_data'          #modify this to test
    train_x_path = os.path.join(rootPath, '2021_09_13_x_metrics.csv')
    train_y_path = os.path.join(rootPath, '2021_09_13_y_labels.csv')
    x = pd.read_csv(train_x_path, delimiter=',', header=None)
    y = (pd.read_csv(train_y_path, delimiter=',', header=None)[0]).to_numpy()
    corr_linkage(x)
    print('done')
    input_metrics = InputMetrics()
    name1 = MetricNames.AUTOCORRELATION
    name2 = MetricNames.COEFFICIENT_OF_VARIANCE
    name3 = MetricNames.FAST_FOURIER_TRANSFORM
    # metric1 = InputMetric(name1, x.T[0])
    # print(metric1.get_value(),metric1.get_name())
    # metric2 = InputMetric(name2, x.T[1])
    #ac1	ac2	cov	ff1	ff2	gse	mean	rms	signal_energy	sma	std	zero_cross	y
    metric1 = InputMetric(name1, x[0])
    metric2 = InputMetric(name2, x[2])
    #metric3 = InputMetric(name3, x[3])
    metric4 = InputMetric(MetricNames.GAIT_SPEED_ESTIMATOR, x[5])
    #metric5 = InputMetric(MetricNames.MEAN, x[6])
    #metric6 = InputMetric(MetricNames.STANDARD_DEVIATION, x[10])
    metric7 = InputMetric(MetricNames.ZERO_CROSSING, x[11])
    input_metrics.set_metric(name1, metric1)
    input_metrics.set_metric(name2, metric2)
    #input_metrics.set_metric(name3, metric3)
    input_metrics.set_metric(MetricNames.GAIT_SPEED_ESTIMATOR, metric4)
    #input_metrics.set_metric(MetricNames.MEAN, metric5)
    #input_metrics.set_metric(MetricNames.STANDARD_DEVIATION, metric6)
    input_metrics.set_metric(MetricNames.ZERO_CROSSING, metric7)
    input_metrics.set_labels(y)
    # cv.plot_data(x, y)
    scaled_input_metrics = classifier.scale_input_data(input_metrics)
    print(cross_validate(classifier, scaled_input_metrics))
    print(train_score(classifier, scaled_input_metrics))
    # cv.plot_classification(classifier.get_model(), x_t, y)
    print(f'Runtime: {time.time() - start}')

def train_score(model, input_metrics):
    x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
    model.train_model(x_train, y_train)
    permutation(model.get_model(),x_test,y_test)
    return model.score_model(x_test, y_test)

def permutation(model,x_test,y_test):
    r = permutation_importance(model, x_test, y_test, scoring='neg_mean_squared_error')
    #r=permutation_importance(model, x_test, y_test, n_repeats=30,random_state=0)
    importance = r.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

def cross_validate(model, input_metrics):
    cv_x, names = input_metrics.get_metric_matrix()
    y = input_metrics.get_labels()
    return model.cross_validate(cv_x, y)

def corr_linkage(X):
    feature_names=['ac1','ac2','cov','ff1','ff2','gse','mean','rms','sig','sma','std','zero_c']
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)


    dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, leaf_rotation=90)
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

if __name__ == '__main__':
    main()
