"""
Experiment for Caltech-101 dataset using base RMSProp
"""
from errors import OptionNotSupportedError
from experiments.caltech101.caltech_exp import CaltechExperiment
from experiments.tester import Tester
from training.trainer.rms_trainer import RMSPropTrainer
from training.config.general_config import GeneralConfig
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode
from utils import constants as const


class CaltechExperimentRMSProp(CaltechExperiment):
    """
    Performs experiments over Caltech-101 dataset using RMSProp
    """
    optimizer_name = const.TR_BASE
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input,
                                      self.input_tensor, self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = GeneralConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                            config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL:
            for i in range(5):
                train_conf = MegabatchConfig(60, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        elif train_mode == TrainMode.ACUMULATIVE:
            train_confs = [MegabatchConfig(60, batch_size=128), MegabatchConfig(50, batch_size=128),
                           MegabatchConfig(30, batch_size=128), MegabatchConfig(30, batch_size=128),
                           MegabatchConfig(30, batch_size=128)]
            self.general_config.train_configurations = train_confs
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))
