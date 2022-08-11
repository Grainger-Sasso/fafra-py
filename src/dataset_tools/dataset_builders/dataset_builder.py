from abc import ABC, abstractmethod

from src.dataset_tools.risk_assessment_data.dataset import Dataset
from src.dataset_tools.dataset_builders.dataset_names import DatasetNames


class DatasetBuilder(ABC):
    def __init__(self, dataset_name):
        self.dataset_name: DatasetNames = dataset_name

    def get_dataset_name(self) -> DatasetNames:
        return self.dataset_name

    def segment_data(self, data, epoch_size, sampling_frequency):
        """
        Segments data into epochs of a given duration starting from the beginning of the data
        Data is in shape m x n where m is number of axes and n is number of samples
        :param: data: data to be segmented
        :param epoch_size: duration of epoch to segment data (in seconds)
        :return: data segments of given epoch duration
        """
        total_time = len(data[0])/sampling_frequency
        # Calculate number of segments from given epoch size
        num_of_segs = int(total_time / epoch_size)
        # Check to see if data can be segmented at least one segment of given epoch size
        if num_of_segs > 0:
            data_segments = []
            # Counter for the number of segments to be created
            segment_count = range(0, num_of_segs+1)
            # Create segmentation indices
            seg_ixs = [int(seg * sampling_frequency * epoch_size) for seg in segment_count]
            for seg_num in segment_count:
                if seg_num != segment_count[-1]:
                    data_segments.append(data[:, seg_ixs[seg_num]: seg_ixs[seg_num+1]])
                else:
                    continue
        else:
            raise ValueError(f'Data of total time {str(total_time)}s can not be '
                             f'segmented with given epoch size {str(epoch_size)}s')
        return data_segments

    @abstractmethod
    def build_dataset(self, dataset_path, clinical_demo_path,
                      segment_dataset, epoch_size) -> Dataset:
        pass

