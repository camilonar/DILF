"""
Module for performing experiments over MNIST
"""
from abc import abstractmethod, ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.le_net import LeNet
from etl.data.mnist_data import MnistData
import utils.constants as const


class MnistExperiment(Experiment, ABC):
    """
    Performs experiments over MNIST according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_MNIST
    data_input = None
    neural_net = None

    def _prepare_data_pipeline(self):
        self.data_input = MnistData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = LeNet({'data': self.input_tensor})
