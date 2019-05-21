"""
Module for the configuration of training that is going to be used.
GeneralConfig has the configuration data for the whole training (including the incremental learning)
"""
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode


class GeneralConfig(object):
    """
    General configuration that it's used for the whole training. E.g. if the training is composed of 5 different batches
    of data, then this part of the configuration is used for **ALL** of them
    """

    def __init__(self, train_mode: TrainMode, learning_rate: float,
                 summary_interval=100, check_interval=200, config_name='default', model_name='dataset_default'):
        """
        Creates a GeneralConfig object

        :param train_mode: Indicates the training mode that is going to be used
        :param learning_rate: the learning rate to be used in the training
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
            performed. Must be an integer multiple of summary_interval
        :param config_name: a descriptive name for the training configuration
        :param model_name: a descriptive name for the model
        """
        self.train_mode = train_mode
        self.learn_rate = learning_rate
        self.summary_interval = summary_interval
        self.check_interval = check_interval
        self.config_name = config_name
        self.model_name = model_name
        self.scenario_id = 0
        self.train_configurations = []  # It stores the configurations for each mega batch of training data

    def add_train_conf(self, train_conf: MegabatchConfig):
        """
        Adds a MegabatchConfig object to the list of mega-batch-specific configurations. The objects must be added in
        the same order as how they are going to be used (i.e. the configuration for the first batch must be added first,
        for the second batch must be added second, ...)

        :param train_conf: a configuration for a mega batch of data
        :return: None
        """
        self.train_configurations.append(train_conf)
