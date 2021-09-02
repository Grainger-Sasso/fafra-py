from typing import Dict


class IMUMetadata:
    def __init__(self, metadata, samp_freq, units):
        self.metadata = metadata
        self.sampling_freq = samp_freq
        self.units: Dict[str, str] = units

    def get_metadata(self):
        return self.metadata

    def get_sampling_frequency(self):
        return self.sampling_freq

    def get_units(self):
        return self.units

