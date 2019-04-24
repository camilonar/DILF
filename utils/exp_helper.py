"""
Module with useful functions for performing an experiment
"""
import utils.constants as const
from experiments.experiments import Experiments
from utils.train_modes import TrainMode


def print_config(dataset: str, optimizer: str, checkpoint_key: str, s_interval: int, ckp_interval: int, seed: int,
                 train_mode: TrainMode, train_dirs: [str], validation_dir: str, is_menu: bool = True):
    """
    Prints the configuration selected by the user

    :param dataset: a string representing the dataset that has been configured by the user
    :param optimizer: a string representing the optimizer that has been configured by the user
    :param checkpoint_key: a string representing a checkpoint. Must be None if no checkpoint has been configured
    :param s_interval: the summary interval that has been configured by the user
    :param ckp_interval: the checkpoint interval that has been configured by the user if the dataset doesn't have
        any dataset-specific path.
    :param seed: the seed for random numbers
    :param train_mode: Indicates the training mode that is going to be used
    :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
    :param validation_dir: a string corresponding to the path of the testing data
    :param is_menu: boolean value. If set to True the menu version is used (asks for extra input)
    """
    print("\n--------------------------------------------------------")
    print("Starting experiment with the following configuration:\n")
    print("-Dataset: {}".format(dataset))
    print("-Optimizer: {}".format(optimizer))
    print("-Checkpoint: {}".format(checkpoint_key))
    print("-Summary interval: {} iterations".format(s_interval))
    print("-Checkpoint interval: {} iterations".format(ckp_interval))
    print("-Seed: {}".format(seed))
    print("-The test mode is {}".format(train_mode))
    print("-Training data directories: \n\t{}".format(train_dirs))
    print("-Validation data directory: \n\t{}".format(validation_dir))
    print("--------------------------------------------------------\n")

    if is_menu:
        input("To continue with the experiment press any key...")


def perform_experiment(dataset_name: str, optimizer_name: str, checkpoint_key: str, s_interval: int, ckp_interval: int,
                       seed: int, train_mode: TrainMode, train_dirs: [str], validation_dir: str):
    """
    Prepares and performs the experiment according to the configuration given by the user

    :param dataset_name: a string representing the dataset that has been configured by the user
    :param optimizer_name: a string representing the optimizer that has been configured by the user
    :param checkpoint_key: a string representing a checkpoint. Must be None if no checkpoint has been configured
    :param s_interval: the summary interval that has been configured by the user
    :param ckp_interval: the checkpoint interval that has been configured by the user
    :param seed: the seed for random numbers
    :param train_mode: Indicates the training mode that is going to be used
    :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
    :param validation_dir: a string corresponding to the path of the testing data
    :return: None
    """
    const.SEED = seed
    factory = Experiments.get_experiment(optimizer_name, dataset_name)
    exp = factory(train_dirs, validation_dir, s_interval, ckp_interval, checkpoint_key)

    exp.prepare_all(train_mode)
    exp.execute_experiment()
