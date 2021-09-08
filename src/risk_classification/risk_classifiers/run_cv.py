from src.risk_classification.validation.data_generator import DataGenerator
from src.visualization_tools.classification_visualizer import ClassificationVisualizer
from src.risk_classification.risk_classifiers.svm_risk_classifier.svm_risk_classifier import SVMRiskClassifier
from src.risk_classification.risk_classifiers.knn_risk_classifier.knn_risk_classifier import KNNRiskClassifier
from src.risk_classification.risk_classifiers.lightgbm_risk_classifier.lightgbm_risk_classifier import LightGBMRiskClassifier


def main():
    # Instantiate the classifier
    classifier = SVMRiskClassifier()
    # classifier = KNNRiskClassifier()
    # classifier = LightGBMRiskClassifier({})

    # Read in data/generate data with random seed set
    x, y = DataGenerator().generate_data()
    # Run the classifier with 5-fold cross validation
    classifier_results = classifier.cross_validate(x, y)
    print(classifier_results)
    ClassificationVisualizer().plot_classification(classifier.get_model(), x)



if __name__ == '__main__':
    main()
