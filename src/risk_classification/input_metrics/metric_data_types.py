from enum import Enum


class MetricDataTypes(Enum):
    EXAMPLE = 'example'
    USER_DATA = 'user_data'
    VERTICAL = 'vertical'
    RESULTANT = 'resultant'

    def get_name(self):
        # Self is the member here
        return self.name

    def get_value(self):
        # Self is the member here
        return self.value

    @classmethod
    def get_all_names(cls):
        return [metric_name.name for metric_name in cls]

    @classmethod
    def get_all_values(cls):
        return [metric_name.value for metric_name in cls]
