import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
from pdpbox import pdp, get_dataset, info_plots

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier


class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self, model: Classifier,
                                               input_metrics: InputMetrics, y):
        x_test, names = input_metrics.get_metric_matrix()
        r = permutation_importance(model.get_model(), x_test, y, scoring='accuracy',n_repeats=50)
        importance = r.importances_mean
        y_pos = np.arange(len(importance))
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.xticks(y_pos, names, color='orange', rotation=15, fontweight='bold', fontsize='5', horizontalalignment='right')
        plt.xlabel('feature Metrix', fontweight='bold', color = 'blue', fontsize='5', horizontalalignment='center')
        plt.show()
    def perform_shap_values(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        model.train_model(x_train, y_train)
        m = model.get_model()
        # explain the model's predictions using SHAP
        explainer = shap.KernelExplainer(m.predict, x_test)
        shap_values = explainer.shap_values(x_test)

        # visualize the first prediction's explaination
        cv,name=input_metrics.get_metric_matrix()
        shap.summary_plot(shap_values, x_test,feature_names=name)
        #p=shap.force_plot(explainer.expected_value, shap_values[0:5,:],x_test[0:5,:])
        # p = shap.force_plot(explainer.expected_value, shap_values,x_test, matplotlib = True, show = False)
        # plt.savefig('tmp.svg')
        # plt.close()
        #shap.plots.force(shap_values)

    def perform_partial_dependence_plot_sklearn(self, model: Classifier,
                                               input_metrics: InputMetrics, y):
        '''This is a implementation of sklearn library by using from sklearn.inspection import PartialDependenceDisplay
        The documentation that I used is: https://scikit-learn.org/stable/modules/partial_dependence.html
        
        '''
        x, names = input_metrics.get_metric_matrix()
        
        na=[eachName.get_name() for eachName in names]
        dataframe = pd.DataFrame(x, columns = na)
        y = input_metrics.get_labels()

        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        clf = model.get_model()
        display = PartialDependenceDisplay.from_estimator(clf,dataframe,na)
        #if I directly use PartialDependenceDisplay.from_estimator(clf,x,names,kind="both"), it returns an errow saying the "feature", which is names in this case, has to be a string or iterable,but ours are object, so  n
        plt.show()
    
    
    def perform_partial_dependence_plot_lightGBM(self, model: Classifier,
                                               input_metrics: InputMetrics, y):
        '''The following code is an implementation for library pdb, the link for this library is: https://github.com/SauceCat/PDPbox
        There is a sample jupyternotebook example in this page: https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb
        '''
        x, names = input_metrics.get_metric_matrix()
        
        na=[eachName.get_name() for eachName in names]
        dataframe = pd.DataFrame(x, columns = na)
        y = input_metrics.get_labels()
        

        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        x_train_dp=pd.DataFrame(x_train, columns = names)
        # train model
        clf = model.get_model() 
        for n in na:
            pdp_sex = pdp.pdp_isolate(
                    model=clf, dataset=dataframe, model_features=na, feature=n,
                    predict_kwds={"ignore_gp_model": True}
                )
            figg,axess=pdp.pdp_plot(pdp_sex,n,plot_lines=True)
            plt.show()


        # pdp_sex = pdp.pdp_isolate(
        #         model=clf, dataset=dataframe, model_features=na, feature=na[1],
        #         predict_kwds={"ignore_gp_model": True}
        #     )
        # print(pdp_sex)
        # fig,axes=pdp.pdp_plot(pdp_sex,na[1],plot_lines=True)
