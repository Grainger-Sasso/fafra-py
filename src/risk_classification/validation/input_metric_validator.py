import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as pl

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
        #plt.sca(axarr[0, 1])   
        # shap_plot=[]
        # for i in range(10):
        #     print(i)
        #     r=explainer.expected_value[i]
        #     r=shap_values[i][0,:]
        #     r=temp[i,:]
        #     f=shap.force_plot(explainer.expected_value[i], shap_values[i][0,:],temp[i,:],show=False,matplotlib=True)#.savefig('scratch.png',format = "png",dpi = 150,bbox_inches = 'tight')
        #     shap_plot.append(pl.gcf())

        shap_metrics = {}
        for name, shap_value in zip(name, shap_values):
            shap_metrics[name] = shap_value.tolist()
        dictionary = {'plots': [shap_plot], 'metrics': shap_metrics}
        return dictionary
