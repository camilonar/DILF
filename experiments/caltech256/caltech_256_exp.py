"""
Module for performing experiments over Caltech-101
"""
from abc import abstractmethod, ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.caffe_net import CaffeNet
from etl.data.caltech_256_data import Caltech256Data
import utils.constants as const


class Caltech256Experiment(Experiment, ABC):
    """
    Performs experiments over Caltech-256 according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_CALTECH_256
    data_input = None
    neural_net = None

    def _prepare_data_pipeline(self):
        self.data_input = Caltech256Data(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = CaffeNet({'data': self.input_tensor})
