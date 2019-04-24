"""
Module for reading a dataset stored in TFRecords
"""
from errors import OptionNotSupportedError
from etl.reader import Reader
from utils.train_modes import TrainMode


class TFRecordsReader(Reader):
    """
    Reader for datasets stored in TFRecords. Its only purpouse is to store the training and testing data paths
    """

    def reload_training_data(self):
        self.accumulative_path.append(self.curr_path)

    def __init__(self, train_dirs: [str], validation_dir: str, train_mode: TrainMode):
        super().__init__(train_dirs, validation_dir, train_mode)
        self.accumulative_path = [self.curr_path]
        print("TEST PATH: ", validation_dir)
        print("TRAIN PATHS: ", train_dirs)

    def load_training_data(self):
        if self.train_mode == TrainMode.INCREMENTAL:
            return self.curr_path, None
        elif self.train_mode == TrainMode.ACUMULATIVE:
            return self.accumulative_path, None
        else:
            raise OptionNotSupportedError("The requested Reader class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, self.train_mode))

    def load_test_data(self):
        return self.test_path, None
