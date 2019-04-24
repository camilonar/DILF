"""
Module for the reader of directory-based datasets.
It reads the dataset from a number of directories previously created and gives a list of filenames and labels.
"""
import random
import os
from typing import List

from errors import OptionNotSupportedError
from etl.reader import Reader
from utils import constants as const
from utils.train_modes import TrainMode

valid_ext = [".jpg", ".gif", ".png", ".jpeg"]


class DirectoryReader(Reader):
    """
    Reader for datasets stored in directories
    """

    def __init__(self, train_dirs: [str], validation_dir: str, train_mode: TrainMode):
        super().__init__(train_dirs, validation_dir, train_mode)
        self.categories = sorted(os.listdir(self.curr_path))
        self.val_filenames, self.val_labels = self._find_image_files(validation_dir, self.categories)
        self.train_filenames, self.train_labels = self._find_image_files(self.curr_path, self.categories)

    def load_class_names(self):
        return self.categories

    def load_training_data(self):
        return self.train_filenames, self.train_labels

    def load_test_data(self):
        return self.val_filenames, self.val_labels

    def reload_training_data(self):
        tr_filenames, tr_labels = self._find_image_files(self.curr_path, self.categories)
        if self.train_mode == TrainMode.INCREMENTAL:
            self.train_filenames, self.train_labels = tr_filenames, tr_labels
        elif self.train_mode == TrainMode.ACUMULATIVE:
            self.train_filenames.extend(tr_filenames)
            self.train_labels.extend(tr_labels)
        else:
            raise OptionNotSupportedError("The requested Reader class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, self.train_mode))

    ###############################################################################
    # TensorFlow Inception function (ported to python3)
    # source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py
    @staticmethod
    def _find_image_files(path: str, categories: List[str]):
        """
        Builds a list of all images files and labels in the data set.

        :param path: directory where the images are located. It is expected that the dataset has a structure
            *path/class_name/*images*. For example:
                    101_ObjectCategories/train/accordion/*.jpg (non-incremental)
                    101_ObjectCategories/train/Increment0/accordion/*.jpg (incremental)
        :param categories: list of strings that contain the caltech dataset categories
        :return: a tuple with two lists, the first one represents the paths of images data and the second one is an
            integer that represents the correct category of the image.
        """
        filenames = []
        labels = []
        # load all images in folder path
        for i, category in enumerate(categories):
            print("LOAD CATEGORY", category)
            for f in os.listdir(path + "/" + category):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_ext:
                    continue
                fullpath = os.path.join(path + "/" + category, f)
                filenames.append(fullpath)
                label_curr = i
                labels.append(label_curr)
        shuffled_index = list(range(len(filenames)))
        random.seed(const.SEED)
        random.shuffle(shuffled_index)
        filenames = [filenames[i] for i in shuffled_index]
        labels = [labels[i] for i in shuffled_index]

        print("Number of filenames: {}".format(len(filenames)))
        print("Number of labels: {}".format(len(labels)))
        ncategories = len(categories)
        print(ncategories)

        return filenames, labels
