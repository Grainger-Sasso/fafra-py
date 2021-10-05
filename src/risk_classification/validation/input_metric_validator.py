
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from eli5.sklearn import PermutationImportance as PermutationImportance_lightGBM

class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self,model:'classifier',input_metrics: InputMetrics,y): 
        x_test, names = input_metrics.get_metric_matrix()
        #def permutation(model,x_test,y_test):
        #r = permutation_importance(model, x_test, y_test, scoring='neg_mean_squared_error')
        r = permutation_importance(model, x_test, y, scoring='accuracy',n_repeats=100)
        importance = r.importances_mean
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
    #def perform_lightgbm_permutation_feature_importance(self,model:'classifier',input_metrics: InputMetrics,y): 
    #    x_test, names = input_metrics.get_metric_matrix()
        

