from enum import Enum


class ClassifierMetrics(Enum):
    EXAMPLE = 'example'
    SHAP = 'shap'
    PFI = 'permutation_feature_importance'
    CV = 'cross_validation'
    PDP = 'partial_dependence_plots'
    LIME = 'lime'

    def get_name(self):
        # Self is the member here
        return self.name

    def get_value(self):
        # Self is the member here
        return self.value

    @classmethod
    def get_all_enum_entries(cls):
        return [metric_name for metric_name in cls]

    @classmethod
    def get_all_names(cls):
        return [metric_name.name for metric_name in cls]

    @classmethod
    def get_all_values(cls):
        return [metric_name.value for metric_name in cls]
