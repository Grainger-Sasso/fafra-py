
class RiskClassificationInputMetrics:
    def __init__(self, id, x, y):
        # Unique identifier of data point (trial subject id)
        self.id = id
        # Variable names are cannonical machine learning terms for inputs to training model
        # Derived values from data
        self.x = x
        # Corresponding outcome conditions for x
        self.y = y

    def get_id(self):
        return self.id

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
