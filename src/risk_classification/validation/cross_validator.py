from sklearn.model_selection import cross_validate


class CrossValidator:
    def __init__(self):
        pass

    def cross_val_model(self, model, x, y, cv=None):
        # cv_results: [dict]; "test_score", "train_score", "fit_time", "score_time", "estimator"
        cv_results = cross_validate(model, x, y, cv=cv, return_estimator=False)
        # for cv_result in enumerate(cross_validate(model, x, y, cv=cv, return_estimator=False)):
        #     cv_results.append({
        #     'test_score': cv_result['test_score'],
        #     'train_score': cv_result['train_score'],
        #     'fit_time': cv_result['fit_time'],
        #     'score_time':  cv_result['score_time']
        # })
        #     print(cv_result['fit_time'], cv_result['score_time'])
        return cv_results

