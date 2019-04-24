"""
Module for performing experiments over MNIST
"""
from abc import ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.fashion_mnist_net import FashionMnistNet
from etl.data.fashion_mnist_data import   FashionMnistData
import utils.constants as const


class FashionMnistExperiment(Experiment, ABC):
    """
    Performs experiments over Fashion-MNIST according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_FASHION_MNIST
    data_input = None
    neural_net = None

    def _prepare_data_pipeline(self):
        self.data_input = FashionMnistData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = FashionMnistNet({'data': self.input_tensor})
