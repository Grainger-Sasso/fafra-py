
class RiskClassificationInputMetrics:
    def __init__(self, x, y):

        # Variable names are cannonical machine learning terms for inputs to training model
        # Derived values from data
        self.x = x
        # Corresponding outcome conditions for x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return y
