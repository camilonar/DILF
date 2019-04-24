"""
Module for saving and loading checkpoints
"""
import os
import tensorflow as tf


class Saver(object):
    """
    Saves and loads checkpoints of models
    """

    def __init__(self):
        """
        Creates a Saver object
        """

        self.iteration_variable, self.mega_batch_variable = None, None
        self.time_variable, self.it_from_start_variable = None, None
        self.iteration, self.mega_batch, self.it_from_start, self.time, self.aux_tensor = None, None, None, None, None
        self.saver = None

    def prepare(self):
        """
        Prepares all tensors and variables needed for proper checkpoint saving and loading. This method only prepares
        the variables used for the basic version of checkpoint loading

        :return: None
        """
        self.iteration_variable = tf.get_variable("iteration", shape=[1], initializer=tf.zeros_initializer)
        self.mega_batch_variable = tf.get_variable("megabatch", shape=[1], initializer=tf.zeros_initializer)
        self.it_from_start_variable = tf.get_variable("it_start", shape=[1], initializer=tf.zeros_initializer)
        self.time_variable = tf.get_variable("time", shape=[1], initializer=tf.zeros_initializer)
        self.aux_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
        self.iteration = self.iteration_variable.assign(self.aux_tensor)
        self.mega_batch = self.mega_batch_variable.assign(self.aux_tensor)
        self.it_from_start = self.it_from_start_variable.assign(self.aux_tensor)
        self.time = self.time_variable.assign(self.aux_tensor)

        self._custom_prepare()

        self.saver = tf.train.Saver()

    def maybe_load_model(self, sess, ckp_path: str):
        """
        This method prepares the previously created neural network with the checkpoint data if a checkpoint is
        provided. It also loads any kind of additional Variable that is need for the training (like Data or Optimizer's
        variables).

        :param sess: current session
        :param ckp_path: the checkpoint path if it's required to start the training from a checkpoint. A data path with
            the following structure is expected: *./checkpoints/dataset_name/config_name/checkpoint_name.ckpt*.
            If there is no checkpoint to be loaded then its value should be None.
        :return: if a checkpoint has been successfully loaded then this method returns a tuple containing 4 values:
            the number of the current mega-batch (increment), iteration over the batch, iteration counting from the
            start of the training, and the time that the network has already been trained (counting from the start of
            the mega-batch) in that order. It returns a tuple of zeros if no checkpoint is loaded.
        """
        if not ckp_path:
            print("No checkpoint has been loaded.")
            return 0, 0, 0, 0
        else:
            print("Loading checkpoint from {}.".format(ckp_path))

        self.saver.restore(sess, ckp_path)
        inc, it, it_t, t = sess.run([self.mega_batch_variable, self.iteration_variable,
                                     self.it_from_start_variable, self.time_variable])
        self._custom_model_load(sess)
        print("Loaded checkpoint at iteration {} of increment {}. Total iterations: {}".format(it, inc, it_t))
        return int(inc[0]), int(it[0] + 1), int(it_t[0] + 1), t[0]

    def save_model(self, sess, ckp_dir: str, iteration: int, total_iteration: int, increment: int, curr_time: float,
                   *args, **kwargs):
        """
        Saves all the variables of the model

        :param sess: current session
        :param ckp_dir: the path to the directory where the checkpoints are going to be saved
        :param iteration: the current iteration number over the training data
        :param total_iteration: the current iteration number over the training data, counting from the start of the
            training (that is, from the first batch of mega-batch 0)
        :param increment: the number of the mega-batch
        :param curr_time: the time that has passed since the beginning of the training of the current batch. This time
            must be in seconds
        """
        filename = "model-{}-{}.ckpt".format(increment, total_iteration)
        sess.run(self.mega_batch, feed_dict={self.aux_tensor: [increment]})
        sess.run(self.iteration, feed_dict={self.aux_tensor: [iteration]})
        sess.run(self.it_from_start, feed_dict={self.aux_tensor: [total_iteration]})
        sess.run(self.time, feed_dict={self.aux_tensor: [curr_time]})
        self._custom_model_save(sess, *args, **kwargs)
        save_path = self.saver.save(sess, os.path.join(ckp_dir, filename))
        print("Model saved in path: {}".format(save_path))
        return save_path

    def _custom_model_load(self, sess):
        """
        This is a hook method that may be used by concrete trainers to define custom attributes to be obtained when a
        checkpoint is loaded. This is intended to be used for information that isn't supported in the base Trainer's
        checkpoint load.
        Please note that the checkpoints are saved and restored using the Saver class from Tensorflow, so all the
        information must be loaded from that source. If you need another kind of checkpoint management then you should
        override the _maybe_load_model and _save_model methods.
        This method isn't implemented by default

        :param sess: the current session
        :return: None
        """
        pass

    def _custom_model_save(self, sess, *args, **kwargs):
        """
        This is a hook method that may be used by concrete trainers to define custom attributes to be stored when a
        checkpoint is saved. This is intended to be used for information that isn't supported in the base Trainer's
        checkpoint saver.
        Please note that the checkpoints are saved and restored using the Saver class from Tensorflow, so all the
        information must be loaded from that source. If you need another kind of checkpoint management then you should
        override the _maybe_load_model and _save_model methods.
        This method isn't implemented by default

        :param sess: the current session
        :return: None
        """
        pass

    def _custom_prepare(self):
        """
        This is a hook method that may be used by concrete savers to define custom preparations for the checkpoint
        loading and saving, which may include the definition of additional variables that need to be stored. This
        method isn't implemented by default

        :return: None
        """
        pass

