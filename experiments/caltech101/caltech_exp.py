"""
Module for performing experiments over Caltech-101
"""
from abc import abstractmethod, ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.alex_net import AlexNet
from etl.data.caltech_data import CaltechData
import utils.constants as const


class CaltechExperiment(Experiment, ABC):
    """
    Performs experiments over Caltech-101 according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_CALTECH_101
    data_input = None
    neural_net = None
    input_tensor = None
    output_tensor = None

    def _prepare_data_pipeline(self):
        self.data_input = CaltechData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = AlexNet({'data': self.input_tensor})
