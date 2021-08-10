import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from mlxtend.plotting import plot_learning_curves
from sklearn.decomposition import PCA



class KNNRiskClassifier:
    def __init__(self, N=10, w='distance'):
        self.model: KNN = KNeighborsClassifier(n_neighbors=N, weights=w)
        self.scaler: StandardScaler = preprocessing.StandardScaler()

    def get_model(self)->'KNN':
        return self.model

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def set_model(self, model: 'KNN'):
        self.model = model

    def set_scaler(self, scaler: StandardScaler):
        self.scaler = scaler
    
    def scale_input_data(self, x):
        self.scaler.fit(x)
        return self.scaler.transform(x)

    def scale_train_test_data(self, x_train, x_test):
        # Fit the scaler to the training data
        self.scaler.fit(x_train)
        # Transform the training data
        x_train_t = self.scaler.transform(x_train)
        # Transform the test data
        x_test_t = self.scaler.transform(x_test)
        return x_train_t, x_test_t
    
    def fit_model(self, x: np.ndarray, y: np.ndarray):
        """
        Fits KNN model to input training vectors, x, and target values, y (notation is canonically used)
        :param x: Training vectors
        :type: np.ndarray
        :param y: Target values
        :type: np.ndarray
        :return: Trained model
        :rtype: KNN
        """
        self.model.fit(x, y)

    def split_input_metrics(self, x, y, test_size=0.33, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def make_prediction(self, samples):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

def main():
    classifier = KNNRiskClassifier()
    '''read data file'''
    rootPath = 'D:/carapace/metric_test_data'
    
    train_x_path = os.path.join(rootPath, 'x_data_metrics.csv')
    train_y_path = os.path.join(rootPath, 'y_data_metrics.csv')
    train_x = pd.read_csv(train_x_path, delimiter=',', header=None)
    
    train_y = (pd.read_csv(train_y_path, delimiter=',', header=None)[0]).to_numpy()
    
    #print(train_x,file=open("./o_origin.txt",'a'))
    scaler = classifier.get_scaler()
    x_data_scaled = scaler.fit_transform(train_x)
    #x_data_scaled = pd.DataFrame(x_data_scaled,columns=['0', '1', '2', '3','4','5','6'])
    skf=StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    reports=[]
    for train_index, test_index in skf.split(x_data_scaled, train_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_data_scaled[train_index], x_data_scaled[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        #print(len(x_train), len(x_test), len(y_train), len(y_test))
    #x_train, x_test, y_train, y_test = classifier.split_input_metrics(x_data_scaled, train_y.values.ravel())
        #print(max(x_train[1]),max(x_train[2]))
        #print(x_train,file=open("./x_train.txt",'a'))
        #print(len(x_train),len(y_train))
        
        x_train = (pd.DataFrame(x_train,columns=['0', '1', '2', '3','4','5','6'])).to_numpy()
        classifier.fit_model(x_train, y_train)

        '''plot'''
        value=1.5
        width=0.75
        #X_train2 = pca.fit_transform(x_train)
        plot_decision_regions(x_train,y_train.astype(int),clf=classifier.get_model(),legend=2,feature_index=[1,2],                        #these one will be plotted  
                    filler_feature_values={0: value, 3:value, 4:value, 5:value, 6:value},  #these will be ignored
                    filler_feature_ranges={0: width, 3: width, 4:width, 5:width, 6:width},X_highlight=x_train)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Knn with K='+ str(5))
        plt.show()

        plot_learning_curves(x_train, y_train, x_test, y_test, classifier.get_model())
        plt.show()

        ypred = classifier.make_prediction(x_test)
        #create confusion matrix and see the classification report
        result = confusion_matrix(y_test, ypred)
        print("Confusion Matrix is {}".format(result))
        report = classifier.create_classification_report(y_test, ypred)
        print(report)
        print("Accuracy: ", classifier.score_model(x_test, y_test))
        reports.append(classifier.score_model(x_test, y_test))
    print("final Accuracy: ", sum(reports)/len(reports))



if __name__ ==  '__main__':
    main()