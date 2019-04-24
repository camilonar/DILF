"""
IoC container for the Experiment objects
"""
import utils.constants as const
from experiments.caltech256.caltech_256_exp_rms import Caltech256ExperimentRMSProp
from experiments.caltech101.caltech_exp_rep import CaltechExperimentRep
from experiments.caltech101.caltech_exp_rms import CaltechExperimentRMSProp
from experiments.cifar100.cifar100_exp_rms import Cifar100ExperimentRMSProp

from experiments.cifar10.cifar_exp_rep import CifarExperimentRep
from experiments.cifar10.cifar_exp_rms import CifarExperimentRMSProp
from experiments.fashion_mnist.fashion_mnist_exp_rep import FashionMnistExperimentRep
from experiments.fashion_mnist.fashion_mnist_exp_rms import FashionMnistExperimentRMSProp

from experiments.imagenet.imagenet_exp_rep import ImagenetExperimentRep
from experiments.imagenet.imagenet_exp_rms import ImagenetExperimentRMSProp

from experiments.mnist.mnist_exp_rep import MnistExperimentRep
from experiments.mnist.mnist_exp_rms import MnistExperimentRMSProp


class Experiments(object):
    """
    IoC container for the basic experiments included in the framework, which corresponds to datasets:

        - Caltech-101
        - CIFAR-10
        - CIFAR-100
        - MNIST
        - Tiny Imagenet
        - Caltech 256
        - Fashion MNIST

    """
    testers = {const.TR_BASE: {const.DATA_CALTECH_101: CaltechExperimentRMSProp,
                               const.DATA_CIFAR_10: CifarExperimentRMSProp,
                               const.DATA_TINY_IMAGENET: ImagenetExperimentRMSProp,
                               const.DATA_MNIST: MnistExperimentRMSProp,
                               const.DATA_CALTECH_256: Caltech256ExperimentRMSProp,
                               const.DATA_CIFAR_100: Cifar100ExperimentRMSProp,
                               const.DATA_FASHION_MNIST: FashionMnistExperimentRMSProp
                               },
               const.TR_REP: {const.DATA_CALTECH_101: CaltechExperimentRep,
                              const.DATA_CIFAR_10: CifarExperimentRep,
                              const.DATA_TINY_IMAGENET: ImagenetExperimentRep,
                              const.DATA_MNIST: MnistExperimentRep,
                              const.DATA_FASHION_MNIST: FashionMnistExperimentRep}
               }

    @classmethod
    def get_experiment(cls, str_trainer: str, str_dataset: str):
        """
        Gets an Experiment object factory

        :param str_trainer: a string representing the trainer/optimizer
        :param str_dataset: a string representing the dataset
        :return: a Factory provider for the desired trainer and dataset
        :rtype: Factory
        """
        return cls.testers[str_trainer][str_dataset]
