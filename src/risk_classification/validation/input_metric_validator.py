import matplotlib.pyplot as plt
import numpy as np
import shap
import lime
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
import matplotlib.pyplot as pl
from pdpbox import pdp

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier


class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self, model: Classifier,
                                               input_metrics: InputMetrics, y,
                                               show_plot=False):
        x_test, names = input_metrics.get_metric_matrix()
        r = permutation_importance(model.get_model(), x_test, y, scoring='accuracy',n_repeats=50)
        importance = r.importances_mean
        y_pos = np.arange(len(importance))
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        bar_plot = plt.bar([x for x in range(len(importance))], importance)
        #bar_plot.xticks(y_pos, names, color='orange', rotation=15, fontweight='bold', fontsize='5', horizontalalignment='right')
        #bar_plot.xlabel('feature Metrics', fontweight='bold', color='blue', fontsize='5', horizontalalignment='center')
        if show_plot:
            plt.show()
        input_metric_names = list([str(i) for i in input_metrics.get_metrics().keys()])
        pfi_metrics = {}
        for name, pfi_value in zip(input_metric_names, importance):
            pfi_metrics[name] = pfi_value
        dictionary = {'plots': [bar_plot], 'metrics': pfi_metrics}
        return dictionary

    def perform_shap_values(self, model, input_metrics: InputMetrics):
        x_test,name = input_metrics.get_metric_matrix()
        np.random.seed(42)
        print(x_test.shape)
        sample_size=50
        
        x_test=x_test.T
        s=np.zeros((x_test.shape[0],sample_size))
        for i in range(x_test.shape[0]):
            s[i]=np.random.choice(x_test[i],sample_size)
        s=s.T
        #s=shap.sample(x_test,50)#np.random.choice(x_test,size=(5,10))
    
        m = model.get_model()
        # explain the model's predictions using SHAP
        explainer = shap.KernelExplainer(m.predict, s)
        shap_values = explainer.shap_values(s)

        # visualize the first prediction's explaination
        #cv,name = input_metrics.get_metric_matrix()
        shap.summary_plot(shap_values, s, feature_names=name,show=False)
        shap_plot = pl.gcf()

        #p=shap.force_plot(explainer.expected_value, shap_values[0:5,:],x_test[0:5,:])
        # p = shap.force_plot(explainer.expected_value, shap_values,x_test, matplotlib = True, show = False)
        # plt.savefig('tmp.svg')
        # plt.close()
        #shap.plots.force(shap_values)
        input_metric_names = list([str(i) for i in input_metrics.get_metrics().keys()])
        shap_metrics = {}
        for name, shap_value in zip(input_metric_names, shap_values):
            shap_metrics[name] = shap_value.tolist()
        dictionary = {'plots': [shap_plot], 'metrics': shap_metrics}
        return dictionary
    def perform_shap_values_gbm(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        #model.train_model(x_train, y_train)
        m = model.get_model()
        m.params['objective'] = 'binary'
        # explain the model's predictions using SHAP
        explainer = shap.TreeExplainer(m)
        np.random.seed(42)
        s=shap.sample(x_test,50)
        shap_values = explainer.shap_values(s)

        # visualize the first prediction's explaination
        name = list([str(i) for i in input_metrics.get_metrics().keys()])
        # fig, axarr = plt.subplots(2, 3)
        # plt.sca(axarr[0, 0])   
        shap.summary_plot(shap_values, s,feature_names=name,show=False)
        shap_plot = pl.gcf()
        temp=np.array([np.array(xi) for xi in x_test])

        shap_metrics = {}
        for name, shap_value in zip(name, shap_values):
            shap_metrics[name] = shap_value.tolist()
        dictionary = {'plots': [shap_plot], 'metrics': shap_metrics}
        return dictionary
    def perform_partial_dependence_plot_knn(self, model: Classifier,
                                                input_metrics: InputMetrics):
        '''This is a implementation of sklearn library by using from sklearn.inspection import PartialDependenceDisplay
        The documentation that I used is: https://scikit-learn.org/stable/modules/partial_dependence.html
    
        '''
        x, names = input_metrics.get_metric_matrix()
    
        na=[eachName.get_name() for eachName in names]
        dataframe = pd.DataFrame(x, columns = na)
        #y = input_metrics.get_labels()
    
        #x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        clf = model.get_model()
        display = PartialDependenceDisplay.from_estimator(clf,dataframe,na,random_state=42)
        #if I directly use PartialDependenceDisplay.from_estimator(clf,x,names,kind="both"), it returns an errow saying the "feature", which is names in this case, has to be a string or iterable,but ours are object, so  n
        #plt.show()
        pdp_plot = pl.gcf()
        pdp_metrics = {}
        o=1
        for name, pdp_value in zip(na, display.pd_results):
            print(type(pdp_value),pdp_value)
            pdp_metrics[name] = [{p:pdp_value[p][0].tolist()} for p in pdp_value]#a dic with average and value as key
        dictionary = {'plots': [pdp_plot], 'metrics': pdp_metrics}
        return dictionary
    
    
    def perform_partial_dependence_plot_lightGBM(self, model: Classifier,
                                                input_metrics: InputMetrics):
        '''The following code is an implementation for library pdb, the link for this library is: https://github.com/SauceCat/PDPbox
        There is a sample jupyternotebook example in this page: https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb
        '''
        x, names = input_metrics.get_metric_matrix()
    
        na=[eachName.get_name() for eachName in names]
        dataframe = pd.DataFrame(x, columns = na)
    
        # train model
        clf = model.get_model()
        index_list=[(i,j) for i in range(2) for j in range(5)]
        
        i=0
        pdp_plot={}
        pdp_metrics={}
        for n in na:
            pdp_sex = pdp.pdp_isolate(
                    model=clf, dataset=dataframe, model_features=na, feature=n,
                    predict_kwds={"ignore_gp_model": True}
                )
            #figg,axess=pdp.pdp_plot(pdp_sex,n,plot_lines=True)
            figg,axess=pdp.pdp_plot(pdp_sex,n,plot_lines=True)
            pdp_plot[n]=pl.gcf()
            pdp_metrics[n]=pdp_sex.pdp
            
        for name, pdp_value in zip(na, pdp_metrics):
            print(type(pdp_metrics[pdp_value]),pdp_metrics[pdp_value])
            pdp_metrics[name] = pdp_metrics[pdp_value].tolist()#[{p:pdp_metrics[pdp_value][p].tolist()} for p in pdp_metrics[pdp_value]]#a dic with average and value as key
        dictionary = {'plots': pdp_plot, 'metrics': pdp_metrics}
        return dictionary


    '''sample the x_test with size: sample size'''
    def sample_x_test(self,x_test,sample_size):
        x_test=x_test.T
        s=np.zeros((x_test.shape[0],sample_size))
        for i in range(x_test.shape[0]):
            s[i]=np.random.choice(x_test[i],sample_size)
        s=s.T
        return s
    def perform_lime(self, model, input_metrics: InputMetrics, value):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        cv, name = input_metrics.get_metric_matrix()

        model.train_model(x_train, y_train)
        m = model.get_model()

        names = []  # without this, won't get feature names
        for i in name:
            names.append(i.get_value())

        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=names, discretize_continuous=True)

        exp = explainer.explain_instance(x_test[value], m.predict_proba, top_labels=1)
        #exp.show_in_notebook(show_table=True, show_all=False).display()

        a = exp.as_html(show_table=True, show_all=False)
        # with open("KNNdata2.html", "w") as file:
        #     file.write(a)
        lime_metrics = {}
        lime_score_dict=a.score
        for name, lime_value in zip(name, lime_score_dict):
            lime_metrics[name] = lime_score_dict[lime_value].tolist()
        dictionary = {'plots': [a], 'metrics': lime_metrics}
        return dictionary