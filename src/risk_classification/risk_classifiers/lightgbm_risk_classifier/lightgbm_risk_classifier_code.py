from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import lightgbm

class LightGBMRiskClassifier:
    def __init__(self):
        self.model = lightgbm.LGBMClassifier()
        self.scaler = preprocessing.StandardScaler()

    def get_model(self):
        return self.model

    def set_model(self, model = lightgbm.LGBMClassifier()):
        self.model = model

    def get_scaler(self):
        return self.scaler

    def set_scaler(self, scaler = preprocessing.StandardScaler()):
        self.scaler = scaler

    def scale_input_data(self, x):

        self.scaler.fit(x)
        return self.scaler.transform(x)


