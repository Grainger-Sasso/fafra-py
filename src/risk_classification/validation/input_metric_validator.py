import shap
import lime
import matplotlib.pyplot as plt
import numpy as np
import shap
import lime
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from pdpbox import pdp

from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.risk_classifiers.classifier import Classifier


class InputMetricValidator:

    def __init__(self):
        pass

    def perform_permutation_feature_importance(self, model: Classifier,
                                               input_metrics: InputMetrics,
                                               show_plot=False):
        x_test, names = input_metrics.get_metric_matrix()
        y = input_metrics.get_labels()
        r = permutation_importance(model.get_model(), x_test, y, scoring='accuracy',n_repeats=50)
        importance = r.importances_mean
        y_pos = np.arange(len(importance))
        # summarize feature importance
        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))
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
        #print(x_test.shape)
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
        shap.summary_plot(shap_values, s, feature_names=name,show=False)
        shap_plot = pl.gcf()

        input_metric_names = list([str(i) for i in input_metrics.get_metrics().keys()])
        shap_metrics = {}
        for name, shap_value in zip(input_metric_names, shap_values):
            shap_metrics[name] = shap_value.tolist()
        dictionary = {'plots': [shap_plot], 'metrics': shap_metrics}
        return dictionary
    def perform_shap_values_gbm(self, model, input_metrics: InputMetrics):
        x_train, x_test, y_train, y_test = model.split_input_metrics(input_metrics)
        # train model
        m = model.get_model()
        m.params['objective'] = 'binary'
        # explain the model's predictions using SHAP
        explainer = shap.TreeExplainer(m)
        np.random.seed(42)
        s=shap.sample(x_test,50)
        shap_values = explainer.shap_values(s)

        # visualize the first prediction's explaination
        name = list([str(i) for i in input_metrics.get_metrics().keys()])
        # name = m.feature_name()
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
    
        # na=[str(eachName.get_name()) for eachName in names]
        dataframe = pd.DataFrame(x, columns = names)
        
        # train model
        clf = model.get_model()

        fig0, axs0 = plt.subplots(5, 2)
        display_full = PartialDependenceDisplay.from_estimator(clf,dataframe,names,random_state=42,ax=axs0,kind='both')
        pl.clf()

        fig, axs = plt.subplots(5, 2)
        fig.tight_layout()
        display = PartialDependenceDisplay.from_estimator(clf,dataframe,names,random_state=42,ax=axs)
        pdp_plot = pl.gcf()
        pdp_metrics = {}
        col=0
        row=0
        j=0

        for name, name_idx,pdp_value in zip(names, [u for u in range(len(names))],display_full.pd_results):
            var_dict={}
            pdp_metrics[name] = [{p:pdp_value[p][0].tolist()} for p in pdp_value]#a dic with average and value as key
            var_list=np.var(pdp_metrics[name][1]["individual"],axis=0)
            for i in range(len(pdp_metrics[name][2]["values"])):
                var_dict[pdp_metrics[name][2]["values"][i]]={"0":var_list[i]}
            pdp_metrics[name].append({'variance':var_dict})
            var_dict=pd.DataFrame.from_dict(var_dict)
            axs[row][col].set_ylim(0,0.8)
            axs[row][col].plot(var_dict.T, color='r')
            axs[row][col].autoscale()
            j=j+1
            col=j%2
            if col==0:
                row+=1



        dictionary = {'plots': [pdp_plot], 'metrics': pdp_metrics}
        return dictionary
    
    
    def perform_partial_dependence_plot_lightGBM(self, model: Classifier,
                                                input_metrics: InputMetrics):
        '''The following code is an implementation for library pdb, the link for this library is: https://github.com/SauceCat/PDPbox
        There is a sample jupyternotebook example in this page: https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb
        '''
        x, names = input_metrics.get_metric_matrix()

        # na=[eachName.get_name() for eachName in names]
        dataframe = pd.DataFrame(x, columns=names)

        # train model
        clf = model.get_model()

        pdp_plot={}
        pdp_metrics={}


        for n in names:
            pdp_sex = pdp.pdp_isolate(
                    model=clf, dataset=dataframe, model_features=names, feature=n,
                    predict_kwds={"ignore_gp_model": True}
                )
            #len o fpdp_plot_data is invalid, since it is pdp isolate objects
            figg,axess=pdp.pdp_plot(pdp_sex,n,plot_lines=True,x_quantile=True)
            axess['pdp_ax'].set_ylim(-1,1)
            var_dict={}
            for name in pdp_sex.ice_lines:
                pdp_values=pdp_sex.ice_lines[name]
                var_dict[str(pdp_values.name)]={"0":pdp_values.var(ddof=0)}#convert to string is really important here
            var_dict_df=pd.DataFrame.from_dict(var_dict)
            axess['pdp_ax'].plot(var_dict_df.T,marker='o', color='r')
            pdp_plot[n]=pl.gcf()
            pdp_metrics[n]=[pdp_sex.ice_lines.to_dict(),pdp_sex.pdp.tolist(),var_dict]

        for name, pdp_value in zip(names, pdp_metrics):
            pdp_metrics[name] = pdp_metrics[pdp_value]#[{p:pdp_metrics[pdp_value][p].tolist()} for p in pdp_metrics[pdp_value]]#a dic with average and value as key
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
        cv, names = input_metrics.get_metric_matrix()

        # names=[str(eachName.get_name()) for eachName in name]
        dataframe = pd.DataFrame(x_test, columns = names)
        m = model.get_model()

        # names = []  # without this, won't get feature names
        # for i in origianl:
        #     names.append(str(i.get_value()))

        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=names, random_state=42,discretize_continuous=True,mode='classification')

        #exp = explainer.explain_instance(x_test[0], m.predict_proba, top_labels=1)
        exp = explainer.explain_instance(x_test[value], m.predict_proba, num_features=len(names),labels=[0,1])

        a = exp.as_html(show_table=True, show_all=False)
        
        lime_metrics = {}
        lime_score_dict={value:exp.score}
        for name, lime_value in zip(names, lime_score_dict):
            lime_metrics[value] = lime_score_dict[lime_value].tolist()
        dictionary = {'plots': [a], 'metrics': lime_metrics}
        return dictionary
