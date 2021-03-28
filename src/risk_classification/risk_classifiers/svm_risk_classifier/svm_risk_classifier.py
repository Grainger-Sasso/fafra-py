import numpy as np
import cvxopt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


class SVMRiskClassifier:
    def __init__(self):
        pass

    def classify(self, x: np.array, y: np.array):
        model = SVC(kernel='linear', C=1E10)
        return model.fit(x, y)

