from typing import Any


class ClinicalDemographicData:
    def __init__(self, id: str, age: float, sex: str, faller_status: Any,
                 height: float, trial: Any, name='', weight=0.0, other=None):
        self.id = id
        self.age = age
        self.sex = sex
        self.faller_status = faller_status
        # Height in cm
        self.height = height
        self.trial = trial
        self.name = name
        # Wight in kg
        self.weight = weight
        self.other = other

    def get_id(self):
        return self.id

    def get_age(self):
        return self.age

    def get_sex(self):
        return self.sex

    def get_faller_status(self):
        return self.faller_status

    def get_height(self):
        return self.height

    def get_trial(self):
        return self.trial

    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight

    def get_other(self):
        return self.other
