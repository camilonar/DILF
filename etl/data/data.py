"""
Module for data input pipelines.
Features:
1. Can load training and testing data separately
2. Adaptable for multiple mega-batches (changing the mega-batch and reloading data). This is useful for incremental
training
3. The pipeline retrieves tensors that can be feed to the training module as needed
"""
from abc import abstractmethod, ABC
from etl.reader import Reader
from training.config.general_config import GeneralConfig


class Data(ABC):
    """
    This class acts as an interface for data pipelines.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines
    """

    def __init__(self, general_config: GeneralConfig, data_reader: Reader,
                 image_shape: tuple, buffer_size=128):
        """
        Creates a Data pipeline object for a dataset composed of images. It also sets the current configuration for
        training as the configuration for the first mega-batch.

        This must be called by the constructors of the subclasses.

        :param general_config: the configuration for the whole training
        :param data_reader: the corresponding Reader of the data of the dataset
        :param image_shape: a tuple indicating which shape should the output image have in the format
                (image_height, image_width, image_depth)
        :param buffer_size: the size of the buffer for various operations (such as shuffling)
        """
        self.general_config = general_config
        self.curr_config = self.general_config.train_configurations[0]
        self.data_reader = data_reader
        self.image_shape = image_shape
        self.buffer_size = buffer_size

    def build_train_data_tensor(self, shuffle=True, augmentation=False, skip_count=0):
        """
        Builds the training data tensor

        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :param skip_count: number of elements to be skipped from the Dataset. If the dataset.batch is applied, then each
            batch is treated as 1 element, so in that case, skip_count is the number of batches to be skipped from the
            Dataset.
        :return: a tuple of an Iterator and two Tensors, where the first value is an Iterator for the
            data, the second value is the tensor of training data and the third value is the tensor with the
            corresponding labels of the data. NOTE: the Iterator must be initialized before the training and label data
            tensors can be used to feed data into a model
        """
        reader_data = self.data_reader.load_training_data()
        return self._build_generic_data_tensor(reader_data, shuffle, augmentation, testing=False,
                                               skip_count=skip_count)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        """
        Builds the test data tensor

        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :return: a tuple of an Iterator and two Tensors, where the first value is an Iterator for the
            data, the second value is the tensor of test data and the third value is the tensor with the corresponding
            labels of the data. NOTE: the Iterator must be initialized before the testing and label data tensors can be
            used to feed data into a model
        """
        reader_data = self.data_reader.load_test_data()
        return self._build_generic_data_tensor(reader_data, shuffle, augmentation, testing=True)

    @abstractmethod
    def _build_generic_data_tensor(self, reader_data, shuffle, augmentation, testing, skip_count=0):
        """
        It creates a generic data tensor with its respective iterator

        :param reader_data: a tuple containing the data obtained from the Reader
        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :param testing: indicates if the data is for testing or not
        :param skip_count: number of elements to be skipped from the Dataset. If the dataset.batch is applied, then each
            batch is treated as 1 element, so in that case, skip_count is the number of batches to be skipped from the
            Dataset.
        :return: a tuple of an Iterator and two Tensors, where the first value is an Iterator for the
            data, the second value is the tensor of test data and the third value is the tensor with the corresponding
            labels of the data. NOTE: the Iterator must be initialized before the testing and label data tensors can be
            used to feed data into a model
        """
        raise NotImplementedError("The subclass hasn't implemented the _build_generic_data_tensor method")

    def change_dataset_part(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.

        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        print("Changing dataset megabatch to megabatch {} in the Data object...".format(index))
        if not self.general_config.train_configurations[index] is self.curr_config:
            self.data_reader.change_dataset_megabatch(index)
            self.curr_config = self.general_config.train_configurations[index]
            self.close()
        else:
            print("The dataset megabatch hasn't been changed because the requested megabatch is the current megabatch")

    def prepare_basic_dataset(self, dataset, cache=False, shuffle=False, batch=True, repeat=False, skip_count=0,
                              shuffle_seed=None):
        """
        Helper function that applies simple and common operations over a dataset such as shuffling, batching (according
        to the current MegabatchConfig), repeating (for multiple epochs), skipping data and using cache

        :param dataset: the dataset to be transformed
        :param cache: whether or not the dataset should be cached in RAM
        :param shuffle: whether or not the dataset should be shuffled
        :param batch: wheter or not the dataset should be batched
        :param repeat: whether or not multiple epochs should be used. If True, the dataset is repeated according to
                as specified in the current MegabatchConfig
        :param skip_count: number of elements to be skipped from the Dataset. If batch=True, then each
            batch is treated as 1 element, so in that case, skip_count is the number of batches to be skipped from the
            Dataset.
        :param shuffle_seed: the seed for the shuffling operation
        :return: a processed dataset
        :rtype: tf.Dataset
        """
        if cache:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=shuffle_seed)

        if batch:
            dataset = dataset.batch(self.curr_config.batch_size)

        if not repeat:
            dataset = dataset.repeat(self.curr_config.epochs)
        print("Skipping {} data".format(skip_count))
        dataset = dataset.skip(skip_count)
        return dataset

    def __del__(self):
        """
        Destroys the object and closes the pipeline

        :return: None
        """
        self.close()

    @abstractmethod
    def close(self):
        """
        Closes the pipeline

        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the close method")
