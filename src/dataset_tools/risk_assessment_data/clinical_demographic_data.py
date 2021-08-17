class ClinicalDemographicData:
    def __init__(self, id: str, age: float, sex: str, faller_status: bool, height: float):
        self.id = id
        self.age = age
        self.sex = sex
        self.faller_status = faller_status
        self.height = height

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
