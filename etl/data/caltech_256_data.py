"""
Module for the data pipeline of Caltech-256 dataset
"""

import tensorflow as tf

from etl.reader.directory_reader import DirectoryReader
from etl.data import Data


class Caltech256Data(Data):
    """
    Data pipeline for Caltech-256 dataset
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: [str],
                 buffer_size=1000,
                 image_height=224,
                 image_width=224):

        print("Loading caltech data...")
        my_caltech = DirectoryReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_caltech, (image_height, image_width, 3), buffer_size=buffer_size)
        self.data_reader.check_if_data_exists()

    def _build_generic_data_tensor(self, reader_data, shuffle, augmentation, testing, skip_count=0):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """
        number_of_classes = 256
        num_of_channels = 3

        def load_images(single_path, single_target):
            """
            Maps the paths and labels with the corresponding Tensors that are going to be used as Input for the training

            :param single_path: a path to an image of .jpeg type
            :param single_target: a number that corresponds with the label of the sample
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
                the label.
            """
            # one hot encode the target
            single_target = tf.cast(tf.subtract(single_target, tf.constant(1)), tf.int32)
            single_target = tf.one_hot(single_target, depth=number_of_classes)

            # load the jpg image according to path
            file_content = tf.read_file(single_path)
            single_image = tf.image.decode_jpeg(file_content, channels=num_of_channels)

            # convert to [0, 1]
            single_image = tf.image.convert_image_dtype(single_image,
                                                        dtype=tf.float32,
                                                        saturate=True)

            single_image = tf.image.resize_images(single_image, [self.image_shape[0], self.image_shape[1]])

            # Data Augmentation
            if augmentation:
                single_image = tf.image.resize_image_with_crop_or_pad(single_image, self.image_shape[0] + 4,
                                                                      self.image_shape[1] + 4)
                single_image = tf.random_crop(single_image, [self.image_shape[0], self.image_shape[1], num_of_channels])
                single_image = tf.image.random_flip_left_right(single_image)

                single_image = tf.image.per_image_standardization(single_image)
            return single_image, single_target

        # Creates the dataset
        filenames = tf.constant(reader_data[0])
        labels = tf.constant(reader_data[1])

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(load_images, num_parallel_calls=8)
        # The dataset is not shuffled here, but in Reader
        dataset = self.prepare_basic_dataset(dataset, repeat=testing, skip_count=skip_count)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()

        return iterator, images_batch, target_batch

    def close(self):
        return
