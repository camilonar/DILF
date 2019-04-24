"""
Module for the data pipeline of MNIST dataset.
"""
import tensorflow as tf

from etl.data import Data
import utils.constants as const
from etl.reader.tfrecords_reader import TFRecordsReader


class FashionMnistData(Data):
    """
    Data pipeline for Fashion-MNIST dataset
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 buffer_size=60000,
                 image_height=28,
                 image_width=28):
        print("Loading fashion mnist data...")
        my_f_mnist = TFRecordsReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_f_mnist, (image_height, image_width, 1), buffer_size=buffer_size)
        self.data_reader.check_if_data_exists()

    def _build_generic_data_tensor(self, reader_data, shuffle, augmentation, testing, skip_count=0):
        """
         Creates the input pipeline and performs some preprocessing.
        """

        def parser(serialized_example):
            """
            Parses a single tf.Example into image and label tensors.

            :param serialized_example: serialized example in tfrecord type
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
                the label.
            """
            number_of_classes = 10
            image_height = 28
            image_width = 28

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string)
                })

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape((image_width * image_height))
            # Reshape from [depth * height * width] to [depth, height, width].

            image = tf.cast(
                tf.transpose(tf.reshape(image, [1, image_height, image_width]), [1, 2, 0]),
                tf.float32)

            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True) * (1 / 255.0)

            image = tf.image.resize_images(image, [self.image_shape[0], self.image_shape[1]])

            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=number_of_classes)

            return image, label

        # Creates the dataset
        filenames = reader_data[0]
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=len(self.general_config.train_configurations))
        dataset = dataset.map(parser, num_parallel_calls=8)
        dataset = self.prepare_basic_dataset(dataset, shuffle=shuffle, cache=True, repeat=testing,
                                             skip_count=skip_count, shuffle_seed=const.SEED)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()
        return iterator, images_batch, target_batch

    def close(self):
        pass
