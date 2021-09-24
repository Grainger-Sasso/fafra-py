import time
from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

def main():
    start = time.time()
    # Instantiate the classifier
    # classifier = SVMRiskClassifier()
    classifier = KNNRiskClassifier()
    #classifier = LightGBMRiskClassifier({})

    # Read in data/generate data with random seed set
    #dg = DataGenerator()
    cv = ClassificationVisualizer()
    #x, y = dg.generate_data()

    rootPath = r'2021_09_13' 
    train_x_path = os.path.join(rootPath, '2021_09_13_x_metrics.csv')
    train_y_path = os.path.join(rootPath, '2021_09_13_y_labels.csv')
    x = pd.read_csv(train_x_path, delimiter=',', header=None)
    y = (pd.read_csv(train_y_path, delimiter=',', header=None)[0]).to_numpy()

    input_metrics = InputMetrics()
    name1 = MetricNames.AUTOCORRELATION_FREQUENCY
    name2 = MetricNames.AUTOCORRELATION_MAGNITUDE
    name3 = MetricNames.COEFFICIENT_OF_VARIANCE
    name4 = MetricNames.FAST_FOURIER_TRANSFORM_FREQUENCY
    name5 = MetricNames.FAST_FOURIER_TRANSFORM_MAGNITUDE
    name6 = MetricNames.GAIT_SPEED_ESTIMATOR
    name7 = MetricNames.MEAN
    name8 = MetricNames.ROOT_MEAN_SQUARE
    name9 = MetricNames.SIGNAL_ENERGY
    name10 = MetricNames.SIGNAL_MAGNITUDE_AREA
    name11 = MetricNames.STANDARD_DEVIATION
    name12 = MetricNames.ZERO_CROSSING
    metric1 = InputMetric(name1, x[0])
    metric2 = InputMetric(name2, x[1])
    metric3 = InputMetric(name3, x[2])
    metric4 = InputMetric(name4, x[3])
    metric5 = InputMetric(name5, x[4])
    metric6 = InputMetric(name6, x[5])
    metric7 = InputMetric(name7, x[6])
    metric8 = InputMetric(name8, x[7])
    metric9 = InputMetric(name9, x[8])
    metric10 = InputMetric(name10, x[9])
    metric11 = InputMetric(name11, x[10])
    metric12 = InputMetric(name12, x[11])
    input_metrics.set_metric(name1, metric1)
    input_metrics.set_metric(name2, metric2)
    input_metrics.set_metric(name3, metric3)
    input_metrics.set_metric(name4, metric4)
    input_metrics.set_metric(name5, metric5)
    input_metrics.set_metric(name6, metric6)
    input_metrics.set_metric(name7, metric7)
    input_metrics.set_metric(name8, metric8)
    input_metrics.set_metric(name9, metric9)
    input_metrics.set_metric(name10, metric10)
    input_metrics.set_metric(name11, metric11)
    input_metrics.set_metric(name12, metric12)
    input_metrics.set_labels(y)
    # cv.plot_data(x, y)
    scaled_input_metrics = classifier.scale_input_data(input_metrics)
    print(cross_validate(classifier, scaled_input_metrics))
    print(train_score(classifier, scaled_input_metrics))
    # cv.plot_classification(classifier.get_model(), x_t, y)
    print(f'Runtime: {time.time() - start}')
    correlated_features(x, scaled_input_metrics)

def train_score(model, input_metrics):
    x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
    model.train_model(x_train, y_train)
    return model.score_model(x_test, y_test)

def cross_validate(model, input_metrics):
    cv_x, names = input_metrics.get_metric_matrix()
    y = input_metrics.get_labels()
    return model.cross_validate(cv_x, y)

def correlated_features(X, input_metrics):
    cv_x, names = input_metrics.get_metric_matrix()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(cv_x).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(
    corr_linkage, labels=names, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
