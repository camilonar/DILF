"""
Module with utility functions for directories and paths' management
"""
from datetime import datetime
import os
from training.config.general_config import GeneralConfig


def get_unique_logdir():
    """
    Creates an unique logging directory based on current datetime (year, month, ..., seconds)

    :return: a string with a directory name. The returned value **DOES NOT** have any OS specific separator, e.g.
        it would return 'log_2018-10-04-49-24' and not 'log_2018-10-04-49-24/'
    :rtype: str
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    return 'log_{}'.format(timestamp)


def prepare_directories(config: GeneralConfig):
    """
    Creates and prepares the directories of checkpoints and summaries for TensorBoard

    :param config: the configuration that is going to be used in the whole training
    :return: a tuple containing a path for a checkpoint directory and a path for a summaries directory
    """
    ckpt_path = os.path.join('checkpoints', config.model_name, config.config_name)
    summaries_path = os.path.join('summaries', config.model_name, config.config_name, get_unique_logdir())
    print("\nTo create a session of Tensorboard to visualize the data of this training use the following command: ")
    print("tensorboard --logdir=\"{}\"\n".format(os.path.abspath(summaries_path)))

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.isdir(summaries_path):
        os.makedirs(summaries_path)

    return ckpt_path, summaries_path


def create_full_checkpoint_path(model_name: str, config_name: str, inc_ckp_path: str, root='checkpoints', ext='.ckpt'):
    """
    Creates the full relative path to a checkpoint. It also checks if the path exists

    :param model_name: the name of the dataset corresponding to the checkpoint (e.g. Imagenet)
    :param config_name: the name of the Optimizer corresponding to the checkpoint (e.g. CEAL)
    :param inc_ckp_path: a string representing the mega-batch and iteration corresponding to the checkpoint. It is
        expected to follow the format *"[mega-batch]-[iteration]"*, e.g. "0-50".
    :param root: the root directory where checkpoints are being stored
    :param ext: the extension of checkpoint files
    :return: a tuple containing the generated path (str) and a boolean that says whether or not the generated path is
        a file that actually exists
    """
    filename = "model-" + inc_ckp_path + ext
    ckpt_path = os.path.join(root, model_name, config_name, filename)
    return ckpt_path, os.path.isfile(ckpt_path + ".index") and os.path.isfile(ckpt_path + ".meta")


