"""
Trainer with RMSProp Optimizer
"""
import tensorflow as tf

from training.config.general_config import GeneralConfig
from training.trainer.trainer import Trainer


class RMSPropTrainer(Trainer):
    """
    Trainer with RMSProp Optimizer
    """

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, megabatch: int, iteration: int, total_it: int):
        return sess.run([train_step, loss],
                        feed_dict={tensor_x: image_batch, tensor_y: target_batch,
                                   self.mask_tensor: self.mask_value})

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tf.multiply(tensor_y, self.mask_tensor),
                                               tf.multiply(net_output, self.mask_tensor))

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)
