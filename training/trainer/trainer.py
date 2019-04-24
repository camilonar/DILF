"""
Module for training a neural network.
Features:
    1. It's independent of the dataset, or model used for training
    2. Does all the needed preparations for the training (e.g. creating a session)
"""

import time
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
import numpy as np

from etl.data import Data
from libs.caffe_tensorflow.network import Network
from training.config.general_config import GeneralConfig
import utils.dir_utils as utils
from training.support.saver import Saver
from experiments.tester import Tester


class Trainer(ABC):
    """
    Has the purpose of training a generic net, with a generic configuration. It serves as a base for custom trainers
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 tester: Tester = None, checkpoint: str = None):
        """
        It creates a Trainer object

        :param config: the configuration for the whole training
        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a trainings
        :param tester: a Tester object that is going to perform tests using the metrics defined in it (e.g. measure
            accuracy of the model). Set to None if you don't want to perform tests.
        :param checkpoint: the checkpoint path if it's required to start the training from a checkpoint. A data path
            with the following structure is expected: *./checkpoints/dataset_name/config_net_name/checkpoint_name.ckpt*.
            If there is no checkpoint to be loaded then its value should be None. The default value is None.

        This method must be called by the constructors of the subclasses.
        """
        self.config = config
        self.model = model
        self.pipeline = pipeline
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.checkpoint = checkpoint

        # Creation of tester and saver
        self.tester = tester
        self.saver = Saver()

        # Creation of additional attributes for training
        self.sess = None
        self.ckp_dir, self.summaries_dir = None, None
        self.loss, self.train_step = None, None

        # Creation of the mask for incremental learning
        self.mask_tensor, self.mask_value = None, None

    def __prepare(self):
        """
        It does the preparation for the training. This preparations include:

            - Creates a session (if a default session is already in use, then this method deletes it and replaces with a
            new one)
            - Creates operators needed for summaries for TensorBoard
            - Sets the Optimizer
            - Sets variables and operators needed for saving checkpoints of the training
            - Initializes all the variables

        :return: None
        """
        self.ckp_dir, self.summaries_dir = utils.prepare_directories(self.config)
        print("Preparing training...")

        # Creates the session
        sess = tf.get_default_session()
        if sess:
            print("Closing previous Session...")
            sess.close()
        self.sess = tf.InteractiveSession()

        # Creates loss function and optimizer
        self.mask_value, self.mask_tensor = self._create_mask(self.model.get_output())
        self.loss = self._create_loss(self.tensor_y, self.model.get_output())
        self.train_step = self._create_optimizer(self.config, self.loss, self.model.trainable_variables)

        # Prepares checkpoints and initializes variables
        self.saver.prepare()
        self.tester.prepare()

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        # Loading model for transfer learning, if applies
        self.model.maybe_load_model(self.sess)

        print("Finished preparations for training...")

    def train(self):
        """
        Trains a neural network with the appropriate configuration. It also does the preparations needed for that
        training. It loads a checkpoint if a valid checkpoint path has been given.

        :return: None
        """
        self.__prepare()
        inc, skip_count, iteration, start_time = self.saver.maybe_load_model(self.sess, self.checkpoint)

        for i in range(inc, len(self.config.train_configurations)):
            self.pipeline.change_dataset_part(i)
            writer = self.tester.create_writer(self.summaries_dir, i)
            training_iterator, data_x, data_y = self.pipeline.build_train_data_tensor(skip_count=skip_count)
            self.tensor_y.set_shape(data_y.get_shape())
            self.sess.run(training_iterator.initializer)
            iteration = self.train_megabatch(i, skip_count, iteration, start_time,
                                             self.config.train_configurations[i].ttime,
                                             self.config, writer, data_x, data_y)
            self._post_process_megabatch(i)
            # Reestablishes time and skip_count to zero after the first mega-batch (useful when a checkpoint is loaded)
            start_time, skip_count = 0, 0
            print("Finished training of megabatch {}...".format(i))
            writer.close()

        self.__finish()

    def train_megabatch(self, megabatch: int, iteration: int, total_iteration: int, trained_time: float, ttime: float,
                        config: GeneralConfig, writer: tf.summary.FileWriter,
                        data_x: tf.Tensor, data_y: tf.Tensor):
        """
        Trains a neural network over an increment of data, with the given configuration and model, over the data that is
        provided. It also saves summaries for TensorBoard and creates checkpoints according to the given configuration.
        Note: if both maximum number of epochs and maximum time are set, then the training finishes when any of the
        stop criteria is met

        :param megabatch: the number of the mega-batch
        :param iteration: the current iteration number over the training data. It should be zero if no checkpoint has
            been loaded or if the mega-batch at which the restored checkpoint is differs from the current mega-batch
        :param total_iteration: the current iteration number over the training data, counting from the start of the
            training (that is, from the first batch of mega-batch 0)
        :param trained_time: number of seconds that the network has been trained (counting from the start of the
            mega-batch). It should be zero if no checkpoint has been loaded or if the current mega-batch is different
            than the one that was loaded from a checkpoint
        :param ttime: number of seconds that the model should be trained. If None, then time restrictions are not used
        :param config: the configuration for the whole training
        :param writer: a FileWriter properly configured
        :param data_x: the tensor associated with the training data
        :param data_y: the tensor that has the corresponding labels of the training data
        :return: the current iteration
        :rtype: int
        """
        print("Starting training of megabatch {}...".format(megabatch))
        start_time = time.time()
        i = total_iteration  # Iteration counting from the start of the training
        self.tester.perform_validation(self.sess, i, writer)  # Performs validation at the beginning

        while True:
            try:
                image_batch, target_batch = self.sess.run([data_x, data_y])
                self._update_mask(self.mask_value, target_batch)

                _, loss = self._train_batch(self.sess, image_batch, target_batch, self.tensor_x, self.tensor_y,
                                            self.train_step, self.loss, megabatch, iteration, i)
                curr_time = time.time() + trained_time  # If a checkpoint has been loaded, it should adapt the time
                interval = curr_time - start_time
                self.tester.save_loss(self.sess, loss, i, writer)

                if self.tester and i % config.summary_interval == 0 and not i == total_iteration:
                    print("Performing validation at iteration: {}. Loss is: {}. "
                          "Time is: {}".format(i, loss, interval))
                    self.tester.perform_validation(self.sess, i, writer)
                if i % config.check_interval == 0:
                    self.saver.save_model(self.sess, self.ckp_dir, iteration, i, megabatch, interval)
                if ttime and interval > ttime:
                    print("Finished megabatch {} in {} seconds. You can see the results of the training using "
                          "Tensorboard".format(megabatch, interval))
                    break

            except OutOfRangeError:
                break
            i += 1
            iteration += 1

        self.tester.perform_validation(self.sess, i, writer)  # Performs validation at the end
        return i

    def __finish(self):
        """
        Performs the required operations to finish the training (like closing pipelines)

        :return: None
        """
        print("Finishing training...")
        self.pipeline.close()

    @staticmethod
    def _create_mask(net_output: tf.Tensor):
        """
        Creates a mask for the model with length equal to the number of outputs in net_output. The purpouse of
        this mask is to be used in a training for turning on and off certain outputs of the net.

        :param net_output: a tensor corresponding to the last layer of a neural network
        :return: a tuple: an array of zeroes with shape [n_outputs], where n_outputs is the number of outputs in
        net_output, and a tf.placeholder with the same shape as the mask
        """
        mask = [0 for _ in range(net_output.shape[1])]
        placeholder = tf.placeholder(tf.float32, shape=[net_output.shape[1]])
        return mask, placeholder

    @staticmethod
    def _update_mask(mask, target_batch):
        """
        Updates the current mask for the model output

        :param mask: the current value of the mask used for the model output
        :param target_batch: the batch of data corresponding to the output, as obtained from the data pipeline
        :return: None
        """
        if np.count_nonzero(mask) < len(mask):
            count = np.count_nonzero(target_batch, axis=0)
            for i, c in enumerate(count):
                if c > 0:
                    mask[i] = 1

    @abstractmethod
    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        """
        Creates a loss function

        :param tensor_y: the tensor corresponding to the output of a training
        :param net_output: a tensor corresponding to the last layer of a neural network
        :return: a Tensor corresponding to the loss function
        """
        raise NotImplementedError("The subclass hasn't implemented the _create_loss method")

    @abstractmethod
    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        """
        Creates the Optimizer for the training (e.g. AdaGradOptimizer)

        :param config: the configuration for the Optimizer
        :param loss: a tensor representing the loss function
        :param var_list: a list with the variables that are going to be trained
        :return: a tf.Optimizer
        """
        raise NotImplementedError("The subclass hasn't implemented the _create_optimizer method")

    def _post_process_megabatch(self, megabatch):
        """
        Does some post processing after the training of a megabatch is completed. It isn't implemented in the base
        version, but may be overridden by a subclass that needs to perform changes to a variable or any kind of process
        after the training.

        :param megabatch: the megabatch number
        :return: None
        """
        pass

    @abstractmethod
    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, megabatch: int, iteration: int, total_it: int):
        """
        Trains the current model over a batch of data

        :param sess: the current Session
        :param image_batch: the batch of data corresponding to the input, as obtained from the data pipeline
        :param target_batch: the batch of data corresponding to the output, as obtained from the data pipeline
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a training
        :param train_step: the current tf.Optimizer
        :param loss: a tensor representing the loss function
        :param megabatch: the number of the mega-batch
        :param iteration: the current iteration counting from the start of the mega-batch
        _param total_it: the current iteration counting from the start of the whole training
        :return: a tuple containing the result of the training, i.e. the result of running train_step and loss (in that
            order) over the batch of data using sess as Session
        """
        raise NotImplementedError("The subclass hasn't implemented the _train_batch method")
