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
    x, y = dg.generate_data()
    dg.plot_data(x, y)
    # Run the classifier with 5-fold cross validation
    classifier_results = classifier.cross_validate(x, y)
    print(classifier_results)
    print(f'CV runtime: {time.time() - start}')
    ClassificationVisualizer().plot_classification(classifier.get_model(), x)


if __name__ == '__main__':
    main()
