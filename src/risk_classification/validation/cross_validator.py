from sklearn.model_selection import cross_validate


class CrossValidator:
    def __init__(self):
        pass

    def cross_val_model(self, model, x, y, cv=None):
        # cv_results: [dict]; "test_score", "train_score", "fit_time", "score_time", "estimator"
        cv_results = cross_validate(model, x, y, cv=cv, return_estimator=False)
        return cv_results

