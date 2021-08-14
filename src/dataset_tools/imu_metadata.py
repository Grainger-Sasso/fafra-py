from typing import Dict


class IMUMetadata:
    def __init__(self, samp_freq, units):
        self.sampling_freq = samp_freq
        self.units: Dict[str, str] = units

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_units(self):
        return self.units

