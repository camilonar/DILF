###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  imagenet input pipeline
# Date:         11.2016
#
#

###############################################################################
# NOTE: this code has been modified from the original version of Imanol Schlag
# to be in line with the architecture of this program
#


""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""

import tensorflow as tf

from etl.data import Data
from etl.reader.directory_reader import DirectoryReader


class ImagenetData(Data):
    """
    Creates an input pipeline for Tiny Imagenet ready to be fed into a model.

    memory calculation:
      1 image is 299*299*3*4 bytes = ~1MB
      1024MB RAM = ~1000 images

    empirical memory usage with default config:
      TensorFlow +500MB
      imagenet_utils (loading all paths and labels) +400MB
      build input pipeline and fill queues +2.2GB

    - decodes jpg images
    - scales images into a uniform size
    - shuffles the input if specified
    - builds batches
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 buffer_size=1000,
                 image_height=256,
                 image_width=256):
        print("Loading imagenet data")
        my_imagenet = DirectoryReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_imagenet, (image_height, image_width, 3), buffer_size=buffer_size)
        self.data_reader.check_if_data_exists()

    def _build_generic_data_tensor(self, reader_data, shuffle, augmentation, testing, skip_count=0):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """
        number_of_classes = 200
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
            image = tf.image.decode_jpeg(file_content, channels=num_of_channels)

            # convert to [0, 1]
            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True)

            image = tf.image.resize_images(image, [self.image_shape[0], self.image_shape[1]])

            # Data Augmentation
            if augmentation:
                distorted_image = tf.random_crop(image, [self.image_shape[0], self.image_shape[1], 3])
                # Randomly flip the image horizontally.
                distorted_image = tf.image.random_flip_left_right(distorted_image)
                # Because these operations are not commutative, consider randomizing
                # the order their operation.
                # NOTE: since per_image_standardization zeros the mean and makes
                # the stddev unit, this likely has no effect see tensorflow#1458.
                distorted_image = tf.image.random_brightness(distorted_image,
                                                             max_delta=63)
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=0.2, upper=1.8)
                # Subtract off the mean and divide by the variance of the pixels.
                image = tf.image.per_image_standardization(distorted_image)

                # Set the shapes of tensors.
                image.set_shape([self.image_shape[0], self.image_shape[1], 3])

            return image, single_target

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
        print("Closing Data pipeline...")
        return
