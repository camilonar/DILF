"""
Module for the configuration of training that is going to be used.
MegabatchConfig has the configuration data for a specific increment of data
"""


class MegabatchConfig(object):
    """
    Configuration for training based on a chunk of data. This is used for a specific increment of data
    """

    def __init__(self, epochs: int, ttime: int = None, batch_size=128):
        """
        Creates a MegabatchConfig object

        :param epochs: number of epochs for the training. If None, then the dataset is repeated forever
        :param ttime: number of seconds that the model should be trained. If None, then time restrictions are not used
        :param batch_size: the sizes of the batches that are going to be used in training. This is different from the
            number of instances of each incremental training. E.g. An incremental training may be of a total of 1000
            samples using batches with batch_size 100
        """
        self.epochs = epochs
        self.ttime = ttime
        self.batch_size = batch_size
