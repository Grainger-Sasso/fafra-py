import time
from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.input_metrics.metric_names import MetricNames
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier
from src.risk_classification.validation.input_metric_validator import InputMetricValidator

def main():
    start = time.time()
    # Instantiate the classifier
    #classifier = SVMRiskClassifier()
    #classifier = KNNRiskClassifier()
    classifier = LightGBMRiskClassifier({})

    # Read in data/generate data with random seed set
    dg = DataGenerator()
    cv = ClassificationVisualizer()
    x, y = dg.generate_data()
    input_metrics = InputMetrics()
    name1 = MetricNames.AUTOCORRELATION
    name2 = MetricNames.FAST_FOURIER_TRANSFORM
    metric1 = InputMetric(name1, x.T[0])
    metric2 = InputMetric(name2, x.T[1])
    input_metrics.set_metric(name1, metric1)
    input_metrics.set_metric(name2, metric2)
    input_metrics.set_labels(y)
    # cv.plot_data(x, y)
    scaled_input_metrics = classifier.scale_input_data(input_metrics)
    print(cross_validate(classifier, scaled_input_metrics))
    print(train_score(classifier, scaled_input_metrics))
    # cv.plot_classification(classifier.get_model(), x_t, y)

    validate = InputMetricValidator() # validator instances
    validate.perform_shap_values(classifier, scaled_input_metrics) # shap metric implementation

    print(f'Runtime: {time.time() - start}')
    

def train_score(model, input_metrics):
    x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
    model.train_model(x_train, y_train)
    return model.score_model(x_test, y_test)

def cross_validate(model, input_metrics):
    cv_x, names = input_metrics.get_metric_matrix()
    y = input_metrics.get_labels()
    return model.cross_validate(cv_x, y)

if __name__ == '__main__':
    main()
