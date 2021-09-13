import time
from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier


def main():
    start = time.time()
    # Instantiate the classifier
    classifier = SVMRiskClassifier()
    # classifier = KNNRiskClassifier()
    # classifier = LightGBMRiskClassifier({})


    # Read in data/generate data with random seed set
    dg = DataGenerator()
    cv = ClassificationVisualizer()
    x, y = dg.generate_data()
    # cv.plot_data(x, y)
    x_t = classifier.scale_input_data(x)
    print(train_score(classifier, x_t, y))
    # print(cross_validate(classifier, x_t, y))

    cv.plot_classification(classifier.get_model(), x_t, y)

    # cv.plot_classification(classifier.get_model(), x_t, y)
    print(f'Runtime: {time.time() - start}')

def train_score(model, x, y):
    x_train, x_test, y_train, y_test = model.split_input_metrics(x, y)
    model.train_model(x_train, y_train)
    return model.score_model(x_test, y_test)

def cross_validate(model, x, y):
    return model.cross_validate(x, y)

if __name__ == '__main__':
    main()
