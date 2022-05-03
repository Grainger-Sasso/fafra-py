import os
import pandas as pd
from sklearn.svm import SVC

from src.risk_classification.risk_classifiers.classifier import Classifier


class SVMRiskClassifier(Classifier):
    def __init__(self, kernel='linear', c=1E10):
        model = SVC(kernel=kernel, C=c)
        super().__init__('SVM', model)
        self.params = {'kernel': 'linear', 'C': 1E10}

    def train_model(self, x, y, **kwargs):
        # x_scaled = self.scale_input_data(x)
        self.model.fit(x, y)

    def make_prediction(self, samples, **kwargs):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test, **kwargs):
        return self.model.score(x_test, y_test)

    def cross_validate(self, x, y, folds=5, **kwargs):
        return self.cross_validator.cross_val_model(self.model, x, y, folds)

def main():
    classifier = SVMRiskClassifier()
    # Read in data/generate data with random seed set
    # Run the classifier with 5-fold cross validation
    # Return the results
    rootPath = 'D:/carapace/metric_test_data'
    train_x_path = os.path.join(rootPath, 'x_data_metrics.csv')
    train_y_path = os.path.join(rootPath, 'y_data_metrics.csv')
    train_x = pd.read_csv(train_x_path, delimiter=',', header=None)
    train_y = pd.read_csv(train_y_path, delimiter=',', header=None)
    scaler = classifier.get_scaler()
    x_data_scaled = scaler.fit_transform(train_x)
    x_train, x_test, y_train, y_test = classifier.split_input_metrics(x_data_scaled, train_y.values.ravel())
    print(x_train,file=open("./x_train.txt",'a'))
    classifier.fit_model(x_train, y_train)
    ypred = classifier.make_prediction(x_test)
    #create confusion matrix and see the classification report
    result = confusion_matrix(y_test, ypred)
    print("Confusion Matrix is {}".format(result))
    report = classifier.create_classification_report(y_test, ypred)
    print(report)
    print("Accuracy: ", classifier.score_model(x_test, y_test))



if __name__ ==  '__main__':
    main()
