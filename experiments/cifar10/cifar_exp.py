"""
Module for performing experiments over CIFAR-10
"""
from abc import ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.cifar_tfnet import CifarTFNet
from etl.data.cifar_data import CifarData
import utils.constants as const


class CifarExperiment(Experiment, ABC):
    """
    Performs experiments over CIFAR-10 according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_CIFAR_10
    data_input = None
    neural_net = None

    def _prepare_data_pipeline(self):
        self.data_input = CifarData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = CifarTFNet({'data': self.input_tensor})
