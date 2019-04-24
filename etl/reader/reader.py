"""
This module is used to abstract the reading of the data from disk
Features:
1. Can load training and testing data separately
2. Adaptable for multiple mega-batches (changing the mega-batch and reloading data)
"""
from abc import ABC, abstractmethod
from typing import List
import os

from utils.train_modes import TrainMode


class Reader(ABC):
    """
    Interface for the reading of data (of a dataset) from disk.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines
    """

    def __init__(self, train_paths: List[str], test_path: str, train_mode: TrainMode):
        """
        Creates a Reader object and points the current training data path to the first one of the list provided

        :param train_paths: a list of paths, where each one corresponds with the location of one part of the dataset,
            which means, that it has the location of each one of the mega-batches
        :param test_path: the path where the test/validation data is located
        :param train_mode: Indicates the training mode that is going to be used

        This must be called by the constructors of the subclasses.
        """
        self.test_path = test_path
        self.train_paths = train_paths
        self.train_mode = train_mode
        self.curr_path = self.train_paths[0]

    @abstractmethod
    def load_training_data(self):
        """
        It loads and prepares the training data to be fed to the input pipeline

        :return: a tuple with the filenames of the images and labels of the training data of the current mega-batch
        """
        raise NotImplementedError("The subclass hasn't implemented the load_training_data method")

    @abstractmethod
    def load_test_data(self):
        """
        It loads and prepares the test data to be fed to the input pipeline

        :return: a tuple with the filenames of the images and labels of the test data of the current mega-batch
        """
        raise NotImplementedError("The subclass hasn't implemented the load_test_data method")

    def check_if_data_exists(self):
        """
        Checks if the directories or files with training and test data exists

        :return: None
        :raises Exception: if the data is not found
        """
        aux_tr_paths = self.train_paths
        # This is used for non-incremental trainings (multiple TFRecords or directories correspond to one training)
        if type(aux_tr_paths[0]) == list:
            aux_tr_paths = aux_tr_paths[0]

        for i, path in enumerate(aux_tr_paths):
            if os.path.exists(path):
                print("Train directory for batch {} seems to exist".format(i))
            else:
                raise Exception("Train directory for batch {} doesn't seem to exist".format(i))

        if os.path.exists(self.test_path):
            print("Validation directory seems to exist")
        else:
            raise Exception("Validation directory doesn't seem to exist.")

    def change_dataset_megabatch(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.

        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        print("Changing dataset megabatch to megabatch {} in the Reader object...".format(index))
        self.curr_path = self.train_paths[index]
        self.reload_training_data()

    @abstractmethod
    def reload_training_data(self):
        """
        Reloads the training data. It should be invoked after a change in the training data.
        In case the data is being stored as a class attribute, then that class attribute should be updated within this
        method.

        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the reload_training_data method")
