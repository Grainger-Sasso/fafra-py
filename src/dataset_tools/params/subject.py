
class Subject:

    def __init__(self, identifier: str, age: int, height: int, weight: float, gender: str):
        self.identifier: str = identifier
        self.age: int = age
        self.height: int = height
        self.weight: float = weight
        self.gender: str = gender

    def get_subject_identifier(self):
        return self.identifier

    def get_subject_age(self):
        return self.age

    def get_subject_height(self):
        return self.height

    def get_subject_weight(self):
        return self.weight

    def get_subject_gender(self):
        return self.gender

