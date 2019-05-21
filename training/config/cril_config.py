"""
Module for the configuration of training that is going to be used.
This configuration is specifically tailored for CRIL algorithm and its derivates
(e.g. NIL).
"""
from training.config.general_config import GeneralConfig
from utils.train_modes import TrainMode


class CRILConfig(GeneralConfig):
    """
    General configuration that it's used for the whole training. For the **CRIL** algorithm
    """
    def __init__(self, train_mode: TrainMode, learning_rate: float,
                 summary_interval=100, check_interval=200, config_name='default', model_name='dataset_default',
                 memory_size=10, n_candidates=10, buffer_size=1, ):
        """
        Creates a CRILConfig object

        :param train_mode: Indicates the training mode that is going to be used
        :param learning_rate: the learning rate to be used in the training
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
            performed. Must be an integer multiple of summary_interval
        :param config_name: a descriptive name for the training configuration
        :param model_name: a descriptive name for the model
        :param memory_size: maximum number of representatives per class that are going to be saved. E.g. with
        memory_size=25 and n_classes=10, then at most 25*10=250 representatives are stored in memory at any given
        time, and at most 25 representatives for a single class.
        :param n_candidates:
        :param buffer_size: number of iteratios
        """
        super().__init__(train_mode, learning_rate, summary_interval, check_interval, config_name, model_name)
        self.memory_size = memory_size
        self.n_candidates = n_candidates
        self.buffer_size = buffer_size
