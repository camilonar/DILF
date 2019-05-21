"""
Experiment for Cifar-10 dataset using the proposed representative-selection algorithm
"""
import copy

from errors import OptionNotSupportedError
from experiments.cifar10.cifar_exp import CifarExperiment
from experiments.tester import Tester
from training.config.cril_config import CRILConfig
from training.trainer.rep_trainer import RepresentativesTrainer
from training.config.general_config import GeneralConfig
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode
from utils import constants as const


class CifarExperimentRep(CifarExperiment):
    """
    Performs experiments over Cifar-10 dataset using the proposed representative-selection algorithm
    """
    optimizer_name = const.TR_REP
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = RepresentativesTrainer(self.general_config, self.neural_net, self.data_input,
                                              self.input_tensor, self.output_tensor,
                                              tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = CRILConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                         config_name=str_optimizer, model_name=self.dataset_name,
                                         n_candidates=20, memory_size=50, buffer_size=1)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(100, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))

    def _prepare_scenarios(self, base_config):
        scenarios = None
        scenarios = self._add_scenario(scenarios, base_config, 'Test with 1% of data stored as representatives')
        scenario = copy.copy(base_config)
        scenario.memory_size = 500
        scenarios = self._add_scenario(scenarios, scenario, 'Test with 10% of data stored as representatives')
        return scenarios
