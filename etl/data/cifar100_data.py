"""
Module for the data pipeline of Cifar-10 dataset
"""
import tensorflow as tf

from etl.data import Data
from etl.reader.tfrecords_reader import TFRecordsReader
from utils import constants as const


class Cifar100Data(Data):
    """
    Data pipeline for Cifar-100
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 buffer_size=1000,
                 image_height=224,
                 image_width=224):
        print("Loading Cifar-100 data...")
        my_cifar = TFRecordsReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_cifar, (image_height, image_width, 3), buffer_size=buffer_size)
        self.data_reader.check_if_data_exists()

    def _build_generic_data_tensor(self, reader_data, shuffle, augmentations, testing, skip_count=0):
        """
        Creates the input pipeline and performs some preprocessing.
        """
        number_of_classes = 100
        image_height = 32
        image_width = 32

        def parser(serialized_example):
            """
            Parses a single tf.Example into image and label tensors.

            :param serialized_example: serialized example in tfrecord type
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
                the label.
            """

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                })

            image = tf.decode_raw(features['image'], tf.uint8)
            image.set_shape([3 * image_height * image_width])

            # Reshape from [depth * height * width] to [depth, height, width].
            image = tf.cast(
                tf.transpose(tf.reshape(image, [3, image_height, image_width]), [1, 2, 0]),
                tf.float32)

            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True) * (1 / 255.0)
            # Data Augmentation
            if augmentations:
                distorted_image = tf.random_crop(image, [image_height, image_width, 3])
                # Randomly flip the image horizontally.
                distorted_image = tf.image.random_flip_up_down(distorted_image)
                distorted_image = tf.image.random_flip_left_right(distorted_image)
                # Because these operations are not commutative, consider randomizing
                # the order their operation.
                # NOTE: since per_image_standardization zeros the mean and makes
                # the stddev unit, this likely has no effect see tensorflow#1458.

                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=0.2, upper=1.8)
                # Subtract off the mean and divide by the variance of the pixels.
                image = tf.image.per_image_standardization(distorted_image)

                # Set the shapes of tensors.
                image.set_shape([image_height, image_width, 3])

            image = tf.image.resize_images(image, [self.image_shape[0], self.image_shape[1]])
            image = tf.image.per_image_standardization(image)

            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=number_of_classes)

            return image, label

        # Creates the dataset
        filenames = reader_data[0]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser, num_parallel_calls=8)
        dataset = self.prepare_basic_dataset(dataset, shuffle=shuffle, cache=True, repeat=testing,
                                             skip_count=skip_count, shuffle_seed=const.SEED)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()

        return iterator, images_batch, target_batch

    def close(self):
        pass
