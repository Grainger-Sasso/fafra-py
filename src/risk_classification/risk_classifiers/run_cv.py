import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.datasets import make_hastie_10_2
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingClassifier

from pdpbox import pdp, get_dataset, info_plots

from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.validation.input_metric_validator import InputMetricValidator


def newMain():
    X, y = make_hastie_10_2(random_state=0)
    est =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
    features = ['0', '1','2','3','4','5','6','7','8','9']
    plt.rcParams['figure.figsize'] = 16, 9
    dataframe = pd.DataFrame(X, columns = features)
    pdp_sex = pdp.pdp_isolate(
                model=est, dataset=dataframe, model_features=features, feature='0'
            )
    print(pdp_sex)
    fig, axes = pdp.pdp_plot(pdp_sex, '0', plot_lines=True, frac_to_plot=0.5, plot_pts_dist=True)
    
    #fig, axes = pdp.pdp_plot(pdp_sex, '0')
    #fig.show()

def main():
    start = time.time()
    # Instantiate the classifier
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
    input_metrics = InputMetrics()
    name1 = MetricNames.AUTOCORRELATION
    name2 = MetricNames.COEFFICIENT_OF_VARIANCE
    name3 = MetricNames.FAST_FOURIER_TRANSFORM
    #metric1 = InputMetric(name1, x.T[0])
    # print(metric1.get_value(),metric1.get_name())
    #metric2 = InputMetric(name2, x.T[1])
    #ac1	ac2	cov	ff1	ff2	gse	mean	rms	signal_energy	sma	std	zero_cross	y
    metric1 = InputMetric(name1, x[0])
    metric2 = InputMetric(name2, x[2])
    metric3 = InputMetric(name3, x[3])
    metric4 = InputMetric(MetricNames.GAIT_SPEED_ESTIMATOR, x[5])
    metric5 = InputMetric(MetricNames.MEAN, x[6])
    metric6 = InputMetric(MetricNames.STANDARD_DEVIATION, x[10])
    metric7 = InputMetric(MetricNames.ZERO_CROSSING, x[11])
    input_metrics.set_metric(name1, metric1)
    input_metrics.set_metric(name2, metric2)
    input_metrics.set_metric(name3, metric3)
    input_metrics.set_metric(MetricNames.GAIT_SPEED_ESTIMATOR, metric4)
    input_metrics.set_metric(MetricNames.MEAN, metric5)
    input_metrics.set_metric(MetricNames.STANDARD_DEVIATION, metric6)
    input_metrics.set_metric(MetricNames.ZERO_CROSSING, metric7)
    input_metrics.set_labels(y)
    #cv.corr_linkage(input_metrics)                 #plot cluster for input metrics features
    # cv.plot_data(x, y)
    scaled_input_metrics = classifier.scale_input_data(input_metrics)
    print(cross_validate(classifier, scaled_input_metrics))
    print(train_score(classifier, scaled_input_metrics))
    validate=InputMetricValidator()                             #permutation importance calculation
    validate.perform_permutation_feature_importance(classifier.get_model(),input_metrics,y)
    #validate.perform_permutation_feature_importance(classifier,input_metrics,y)
    # cv.plot_classification(classifier.get_model(), x_t, y)

    validate = InputMetricValidator() # validator instances
    validate.perform_shap_values(classifier, scaled_input_metrics) # shap metric implementation

    print(f'Runtime: {time.time() - start}')
    

def train_score(model, input_metrics):
    x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
    print(x_train)
    print("y",y_train,len(y_train),type(y_train))
    model.train_model(x_train, y_train)
    #permutation(model.get_model(),x_test,y_test)
    return model.score_model(x_test, y_test)


def cross_validate(model, input_metrics):
    cv_x, names = input_metrics.get_metric_matrix()
    y = input_metrics.get_labels()
    return model.cross_validate(cv_x, y)




if __name__ == '__main__':
    #main()
    newMain()
