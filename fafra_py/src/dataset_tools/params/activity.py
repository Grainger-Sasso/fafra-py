
class Activity:

    def __init__(self, code: str, fall: bool, description: str, trials: int, duration: int):
        self.code: str = code
        self.fall: bool = fall
        self.description: str = description
        self.trials: int = trials
        self.duration: int = duration

    def get_code(self):
        return self.code

    def get_fall(self):
        return self.fall

    def get_description(self):
        return self.description

    def get_trials(self):
        return self.trials

    def get_duration(self):
        return self.duration
